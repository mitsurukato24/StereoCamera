#include <iostream>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <chrono>
#include "config.h"

enum Infrared { LEFT = 1, RIGHT = 2 };

int main() try
{
	// ------------------------ Read Camera Parameters --------------------------------
	Config::setParameterFile("./parameters.yml");
	std::string filename = Config::get<std::string>("calibration_file");
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		std::cout << "File can not be opend" << std::endl;
		return -1;
	}
	const int width = (int)fs["image_width"], height = (int)fs["image_height"], fps = 60;
	cv::Size img_size(width, height);

	cv::Mat left_cam_mat, right_cam_mat, left_dist_err, right_dist_err;
	fs["left_camera_matrix"] >> left_cam_mat;
	fs["right_camera_matrix"] >> right_cam_mat;
	fs["left_distortion_error"] >> left_dist_err;
	fs["right_distortion_error"] >> right_dist_err;
	cv::Mat left_R, right_R, left_P, right_P, R, T, Q;
	fs["left_R"] >> left_R;
	fs["right_R"] >> right_R;
	fs["left_P"] >> left_P;
	fs["right_P"] >> right_P;
	fs["R"] >> R;
	fs["T"] >> T;
	fs["Q"] >> Q;
	cv::Rect left_roi, right_roi;
	fs["left_roi"] >> left_roi;
	fs["right_roi"] >> right_roi;
	cv::Mat left_map1, left_map2, right_map1, right_map2;
	fs["left_map1"] >> left_map1;
	fs["left_map2"] >> left_map2;
	fs["right_map1"] >> right_map1;
	fs["right_map2"] >> right_map2;

	// ------------------------------- Pipeline Settings ---------------------------------
	rs2::pipeline pipeline;
	rs2::config rs_cfg;
	rs_cfg.disable_all_streams();
	rs_cfg.enable_stream(RS2_STREAM_INFRARED, 1, width, height, RS2_FORMAT_Y8, fps);
	rs_cfg.enable_stream(RS2_STREAM_INFRARED, 2, width, height, RS2_FORMAT_Y8, fps);
	rs_cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
	rs2::pipeline_profile pipeline_profile = pipeline.start(rs_cfg);
	auto depth_sensor = pipeline_profile.get_device().first<rs2::depth_sensor>();
	depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, Config::get<float>("rs2_emitter"));

	// ------------------------- Stereo Matching Parameters -------------------------------
	cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create();
	int num_disparities = ((width / 8) + 15) & -16;
	bm->setROI1(left_roi);
	bm->setROI2(right_roi);
	bm->setPreFilterCap(Config::get<int>("bm_pre_filter_cap"));
	bm->setBlockSize(Config::get<int>("sad_window_size"));
	bm->setMinDisparity(Config::get<int>("bm_min_disparity"));
	bm->setNumDisparities(num_disparities);
	bm->setTextureThreshold(Config::get<int>("bm_texture_threshold"));
	bm->setUniquenessRatio(Config::get<int>("bm_uniqueness_ratio"));
	bm->setSpeckleWindowSize(Config::get<int>("bm_speckle_window_size"));
	bm->setSpeckleRange(Config::get<int>("bm_speckle_range"));
	bm->setDisp12MaxDiff(Config::get<int>("bm_disp_12_max_diff"));

	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create();
	int sgbm_window_size = Config::get<int>("sgbm_window_size");
	int cn = 1;
	sgbm->setPreFilterCap(Config::get<int>("sgbm_pre_filter_cap"));
	sgbm->setBlockSize(sgbm_window_size);
	sgbm->setP1(8 * cn * sgbm_window_size * sgbm_window_size);
	sgbm->setP2(32 * cn * sgbm_window_size * sgbm_window_size);
	sgbm->setMinDisparity(Config::get<int>("sgbm_min_disparity"));
	sgbm->setNumDisparities(num_disparities);
	sgbm->setUniquenessRatio(Config::get<int>("sgbm_uniqueness_ratio"));
	sgbm->setSpeckleWindowSize(Config::get<int>("sgbm_speckle_window_size"));
	sgbm->setSpeckleRange(Config::get<int>("sgbm_speckle_range"));
	sgbm->setDisp12MaxDiff(Config::get<int>("sgbm_disp_12_max_diff"));
	sgbm->setMode(cv::StereoSGBM::MODE_SGBM);

	int count = 0;
	double mean_bm_time = 0., mean_sgbm_time = 0.;
	rs2::frameset frameset;
	while (true)
	{
		// --------------------------- Get Images -------------------------------------
		if (!pipeline.poll_for_frames(&frameset)) continue;
		rs2::frame left_frame = frameset.get_infrared_frame(LEFT);
		rs2::frame right_frame = frameset.get_infrared_frame(RIGHT);
		rs2::frame color_frame = frameset.get_color_frame();

		cv::Mat left_mat(img_size, CV_8UC1, (void*)left_frame.get_data(), cv::Mat::AUTO_STEP);
		cv::Mat right_mat(img_size, CV_8UC1, (void*)right_frame.get_data(), cv::Mat::AUTO_STEP);
		cv::Mat color_mat(img_size, CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

		// -------------------------- Stereo Rectify ------------------------------------
		cv::remap(left_mat, left_mat, left_map1, left_map2, cv::INTER_LINEAR);
		cv::remap(right_mat, right_mat, right_map1, right_map2, cv::INTER_LINEAR);

		cv::Mat show;
		cv::hconcat(left_mat, right_mat, show);
		cv::imshow("Infrared - Left - Right", show);
		cv::waitKey(1);

		// -------------------------- Compute Disparities -------------------------
		count++;
		bool log_calc_time = false;
		std::chrono::system_clock::time_point start, end;
		start = std::chrono::system_clock::now();
		cv::Mat bm_disp, sgbm_disp;
		bm->compute(left_mat, right_mat, bm_disp);
		end = std::chrono::system_clock::now();
		double bm_time = 0.001 * static_cast<double> (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
		mean_bm_time += bm_time;
		if (log_calc_time)
		{
			printf("BM : %lf ms\n", bm_time);
			printf("Mean BM : %lf ms\n", mean_bm_time / count);
		}
		
		start = std::chrono::system_clock::now();
		sgbm->compute(left_mat, right_mat, sgbm_disp);
		end = std::chrono::system_clock::now();
		double sgbm_time = 0.001 * static_cast<double> (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
		mean_sgbm_time += sgbm_time;
		if (log_calc_time)
		{
			printf("SGBM : %lf ms\n", sgbm_time);
			printf("Mean SGBM : %lf ms\n", mean_sgbm_time / count);
		}

		// ----------------------------- Show Disparities -----------------------------------
		cv::Mat show_bm_disp, show_sgbm_disp, show_disp;
		bm_disp.convertTo(show_bm_disp, CV_8U, 255 / (num_disparities * 16.));
		sgbm_disp.convertTo(show_sgbm_disp, CV_8U, 255 / (num_disparities * 16.));
		cv::applyColorMap(show_bm_disp, show_bm_disp, cv::COLORMAP_JET);
		cv::applyColorMap(show_sgbm_disp, show_sgbm_disp, cv::COLORMAP_JET);
		cv::hconcat(show_bm_disp, show_sgbm_disp, show_disp);
		cv::imshow("show disparity", show_disp);
		cv::waitKey(1);

		// ----------------------------- Compute Depth ---------------------------------
		bm_disp.convertTo(bm_disp, CV_32F, 1.0 / 16.f);
		sgbm_disp.convertTo(sgbm_disp, CV_32F, 1.0 / 16.f);
		cv::Mat_<float> bm_depth(bm_disp.size()), sgbm_depth(sgbm_disp.size());
		float fx = (left_cam_mat.at<double>(0, 0) + right_cam_mat.at<double>(0, 0)) / 2.f;
		float baseline = 50.f;
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				if (bm_disp.at<float>(y, x) > 0.f)
				{
					bm_depth(y, x) = fx * baseline / bm_disp.at<float>(y, x);
				}
				else
				{
					bm_depth(y, x) = std::numeric_limits<float>::max();
				}

				if (sgbm_disp.at<float>(y, x) > 0.f)
				{
					sgbm_depth(y, x) = fx * baseline / sgbm_disp.at<float>(y, x);
				}
				else
				{
					sgbm_depth(y, x) = std::numeric_limits<float>::max();
				}
			}
		}

		cv::Mat_<float> show_depth;
		cv::hconcat(bm_depth, sgbm_depth, show_depth);
		imshow("depth", show_depth);
		cv::waitKey(1);

		// -------------------------- Check center Depth ----------------------------
		std::cout << "bm depth : " << bm_depth(height / 2, width / 2) << "mm" << std::endl;
		std::cout << "sgbm depth : " << sgbm_depth(height / 2, width / 2) << "mm" << std::endl;
	}

	return 0;
}
catch (const rs2::error &e)
{
	std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
	return EXIT_FAILURE;
}
catch (const std::exception& e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}