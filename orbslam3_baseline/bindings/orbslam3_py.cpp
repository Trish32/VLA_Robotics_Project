// Minimal pybind11 surface over ORB_SLAM3::System — just enough to serve the same ZMQ
// `track` contract droidSLAM_monocular/slam_server.py speaks, so ros2_bridge's slam_node
// drives either tracker from a launch-file switch.
//
// Deliberately NOT a general wrapper. Exposing the map, keyframes and atlas would invite
// callers to reach into ORB-SLAM3's threads while they are running; everything here is
// call-and-return on the tracking thread.
//
// Two conventions are load-bearing and both are silent when wrong:
//
//   POSE DIRECTION. `System::TrackRGBD` returns Sophus::SE3f Tcw — WORLD-TO-CAMERA. ROS
//   odometry, and our wire contract, want camera-to-world ("where the camera IS in the
//   map frame"). This inverts once, here, so exactly one place in the codebase owns the
//   convention. Returning Tcw directly yields a trajectory mirrored through the origin
//   that still looks like a plausible walk through a room.
//
//   DEPTH UNITS. ORB-SLAM3 divides depth by `DepthMapFactor` from the settings file. We
//   hand it float32 METRES (the same contract depth_to_metres produces for DROID), so
//   the generated settings must carry DepthMapFactor: 1.0. Any other value rescales the
//   map without raising.
//
// Tracking state is reported rather than interpreted: the caller decides what to publish
// when tracking is lost, matching how slam_server.py already refuses to emit a pose
// before the first keyframe.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>

#include "System.h"

namespace py = pybind11;

namespace {

using ArrayU8 = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using ArrayF32 = py::array_t<float, py::array::c_style | py::array::forcecast>;

cv::Mat ColourToMat(const ArrayU8 &array) {
  auto info = array.request();
  if (info.ndim != 3 || info.shape[2] != 3) {
    throw std::invalid_argument(
        "colour image must be (H, W, 3) uint8; got " + std::to_string(info.ndim) +
        " dimensions");
  }
  // CLONE, do not borrow. An earlier version wrapped numpy's buffer directly to avoid
  // the copy, and it was a use-after-free: ORB_SLAM3::Frame RETAINS the image, so when
  // Python rebound the array on the next iteration the buffer was freed under a live
  // reference. It survived ~700 frames of TUM fr3/walking before SIGBUS, which is the
  // worst way for this class of bug to behave. ~900 KB per frame is nothing next to
  // tracking.
  return cv::Mat(static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]),
                 CV_8UC3, info.ptr)
      .clone();
}

cv::Mat DepthToMat(const ArrayF32 &array, int rows, int cols) {
  auto info = array.request();
  if (info.ndim != 2) {
    throw std::invalid_argument("depth must be (H, W) float32 metres");
  }
  if (info.shape[0] != rows || info.shape[1] != cols) {
    throw std::invalid_argument(
        "depth is not aligned to colour: depth is " + std::to_string(info.shape[0]) +
        "x" + std::to_string(info.shape[1]) + ", colour is " + std::to_string(rows) +
        "x" + std::to_string(cols) +
        ". An unaligned depth stream is a different viewpoint, not a different "
        "resolution — use aligned_depth_to_color.");
  }
  return cv::Mat(rows, cols, CV_32F, info.ptr).clone();   // same reason as the colour
}

py::array_t<double> SE3ToNumpy(const Sophus::SE3f &pose) {
  py::array_t<double> out({4, 4});
  auto view = out.mutable_unchecked<2>();
  const Eigen::Matrix4f matrix = pose.matrix();
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      view(r, c) = static_cast<double>(matrix(r, c));
    }
  }
  return out;
}

class Tracker {
 public:
  Tracker(const std::string &vocabulary, const std::string &settings, bool use_viewer)
      : system_(std::make_unique<ORB_SLAM3::System>(
            vocabulary, settings, ORB_SLAM3::System::RGBD, use_viewer)) {}

  py::dict TrackRGBD(const ArrayU8 &colour, const ArrayF32 &depth, double timestamp) {
    cv::Mat colour_mat = ColourToMat(colour);
    cv::Mat depth_mat = DepthToMat(depth, colour_mat.rows, colour_mat.cols);

    Sophus::SE3f world_to_camera;
    {
      // ORB-SLAM3 runs its own threads and blocks here for the tracking duration.
      // Releasing the GIL keeps the Python server responsive and is required if the
      // caller ever tracks from more than one thread.
      py::gil_scoped_release release;
      world_to_camera = system_->TrackRGBD(colour_mat, depth_mat, timestamp);
    }

    const int state = system_->GetTrackingState();
    const bool ok = state == 2;  // ORB_SLAM3::Tracking::OK

    py::dict reply;
    // Inverted here so exactly one place owns the direction; see the file header.
    reply["camera_to_world"] = SE3ToNumpy(world_to_camera.inverse());
    reply["tracking_state"] = state;
    reply["tracking_lost"] = !ok;
    return reply;
  }

  int state() const { return system_->GetTrackingState(); }

  void Reset() { system_->Reset(); }

  void Shutdown() {
    if (system_) {
      system_->Shutdown();
    }
  }

  ~Tracker() { Shutdown(); }

 private:
  std::unique_ptr<ORB_SLAM3::System> system_;
};

}  // namespace

PYBIND11_MODULE(orbslam3_py, m) {
  m.doc() = "Thin ORB-SLAM3 RGB-D tracker for the VLAProjects SLAM wire contract";

  // ORB_SLAM3::Tracking's states, as module constants. NOT py::enum_<int> --
  // pybind11 requires a real enum type there, and `int` has no underlying_type.
  // Exposing them by name keeps callers from hardcoding integers.
  m.attr("SYSTEM_NOT_READY") = -1;
  m.attr("NO_IMAGES_YET") = 0;
  m.attr("NOT_INITIALIZED") = 1;
  m.attr("OK") = 2;
  m.attr("RECENTLY_LOST") = 3;
  m.attr("LOST") = 4;
  m.attr("OK_KLT") = 5;

  py::class_<Tracker>(m, "Tracker")
      .def(py::init<const std::string &, const std::string &, bool>(),
           py::arg("vocabulary"), py::arg("settings"), py::arg("use_viewer") = false,
           "Load ORBvoc.txt and a settings yaml. The viewer needs a display; the ZMQ "
           "server always runs headless.")
      .def("track_rgbd", &Tracker::TrackRGBD, py::arg("colour"), py::arg("depth"),
           py::arg("timestamp"),
           "colour (H, W, 3) uint8 + depth (H, W) float32 METRES -> "
           "{camera_to_world (4,4), tracking_state, tracking_lost}")
      .def("state", &Tracker::state)
      .def("reset", &Tracker::Reset)
      .def("shutdown", &Tracker::Shutdown);
}
