#include <iostream>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <chrono>

#define ESC_KEY 27
#define SPACE_KEY 32

enum Infrared { LEFT = 1, RIGHT = 2 };


double calculate_average_reprojection_err(
	std::vector<std::vector<cv::Point3f>> &obj_pts,
	std::vector<std::vector<cv::Point2f>> &corner_pts,
	std::vector<cv::Mat> &rvecs, std::vector<cv::Mat> &tvecs,
	cv::Mat camera_mat, cv::Mat dist_coeff
)
{
	double total_err = 0, err;
	int total_pts = 0;
	std::vector<float> reproj_errs;
	std::vector<cv::Point2f> tmp_img_pts;
	reproj_errs.resize(obj_pts.size());
	for (int i = 0; i < (int)obj_pts.size(); i++)
	{
		cv::projectPoints(cv::Mat(obj_pts[i]), rvecs[i], tvecs[i], camera_mat, dist_coeff, tmp_img_pts);
		err = cv::norm(cv::Mat(corner_pts[i]), cv::Mat(tmp_img_pts), cv::NORM_L2);
		int n = (int)obj_pts[i].size();
		reproj_errs[i] = (float)std::sqrt(err*err / n);
		total_err += err*err;
		total_pts += n;
	}
	double total_avg_err = std::sqrt(total_err / total_pts);
	printf("Calibration succeeded, avg reprojection error = %.7f\n", total_avg_err);
	return total_avg_err;
}


// Press SPACE to choose image to use for calibration.
bool trinocular_stereo_calibration(
	rs2::pipeline &pipeline, const cv::Size img_size, const cv::Size board_size, const float chess_size
)
{
	int min_num_frames = 10;
	std::vector<std::vector<cv::Point2f>> left_corner_pts, right_corner_pts, color_corner_pts;  // corner points in image coordinates
	cv::TermCriteria term_criteia(
		cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
		100,  // max number of iterations
		0.00001  // min accuracy
	);
	while (true)
	{
		// -------------------------------- Get Frames --------------------------------------
		rs2::frameset frameset;
		if (!pipeline.poll_for_frames(&frameset)) continue;

		rs2::frame left_frame = frameset.get_infrared_frame(LEFT);
		rs2::frame right_frame = frameset.get_infrared_frame(RIGHT);
		rs2::frame color_frame = frameset.get_color_frame();
		cv::Mat left_mat(img_size, CV_8UC1, (void*)left_frame.get_data(), cv::Mat::AUTO_STEP);
		cv::Mat right_mat(img_size, CV_8UC1, (void*)right_frame.get_data(), cv::Mat::AUTO_STEP);
		cv::Mat color_mat(img_size, CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

		// ------------------------------- Find Chessboard ---------------------------------
		std::vector<cv::Point2f> buf_left_corner_pts, buf_right_corner_pts, buf_color_corner_pts;
		int chess_board_flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
		bool left_found = cv::findChessboardCorners(left_mat, board_size, buf_left_corner_pts, chess_board_flags);
		bool right_found = cv::findChessboardCorners(right_mat, board_size, buf_right_corner_pts, chess_board_flags);
		bool color_found = cv::findChessboardCorners(color_mat, board_size, buf_color_corner_pts, chess_board_flags);

		if (left_found && right_found && color_found)
		{
			// -------------------- Use CornerSubPix To Improve Accuracy ------------------
			cv::Size win_size(11, 11);  // half of search window
			cv::cornerSubPix(left_mat, buf_left_corner_pts, win_size, cv::Size(-1, -1), term_criteia);
			cv::cornerSubPix(right_mat, buf_right_corner_pts, win_size, cv::Size(-1, -1), term_criteia);
			cv::Mat gray;
			cv::cvtColor(color_mat, gray, cv::COLOR_BGR2GRAY);
			cv::cornerSubPix(gray, buf_color_corner_pts, win_size, cv::Size(-1, -1), term_criteia);

			// --------------------------- Draw Chessboard -----------------------------------
			cv::Mat show_left, show_right, show_color;
			cv::cvtColor(left_mat, show_left, cv::COLOR_GRAY2BGR);
			cv::cvtColor(right_mat, show_right, cv::COLOR_GRAY2BGR);
			show_color = color_mat.clone();
			cv::drawChessboardCorners(show_left, board_size, cv::Mat(buf_left_corner_pts), left_found);
			cv::drawChessboardCorners(show_right, board_size, cv::Mat(buf_right_corner_pts), right_found);
			cv::drawChessboardCorners(show_color, board_size, cv::Mat(buf_color_corner_pts), color_found);
			cv::Mat show_calib;
			cv::hconcat(std::vector<cv::Mat>{show_left, show_right, show_color}, show_calib);
			cv::imshow("calibration : " + std::to_string((int)buf_left_corner_pts.size()), show_calib);
			int key = cv::waitKey(1);
			if (key == ESC_KEY) break;
			else if (key == SPACE_KEY)
			{
				cv::imshow("show picked image", show_calib);
				int k = cv::waitKey(0);
				if (k == SPACE_KEY)
				{
					left_corner_pts.push_back(buf_left_corner_pts);
					right_corner_pts.push_back(buf_right_corner_pts);
					color_corner_pts.push_back(buf_color_corner_pts);
				}
			}
		}
		else
		{
			cv::Mat show_img;
			cv::Mat show_left, show_right, show_color;
			cv::cvtColor(left_mat, show_left, cv::COLOR_GRAY2BGR);
			cv::cvtColor(right_mat, show_right, cv::COLOR_GRAY2BGR);
			cv::hconcat(std::vector<cv::Mat>{show_left, show_right, color_mat}, show_img);
			cv::imshow("not found", show_img);
			int key = cv::waitKey(1);
			if (key == ESC_KEY) break;
		}
	}

	if ((int)left_corner_pts.size() >= min_num_frames)
	{
		// ----------------------------- Calibrate Each Camera --------------------------------------
		std::vector<cv::Point3f> left_new_obj_pts, right_new_obj_pts, color_new_obj_pts;
		std::vector<std::vector<cv::Point3f>> obj_pts(1);  // true 3d coordinates of corners
		for (int h = 0; h < board_size.height; h++)
		{
			for (int w = 0; w < board_size.width; w++)
			{
				obj_pts[0].push_back(cv::Point3f(chess_size*w, chess_size*h, 0));
			}
		}
		left_new_obj_pts = obj_pts[0];
		right_new_obj_pts = obj_pts[0];
		color_new_obj_pts = obj_pts[0];
		obj_pts.resize(left_corner_pts.size(), obj_pts[0]); // copy

		std::vector<cv::Mat> left_rvecs, left_tvecs, right_rvecs, right_tvecs, color_rvecs, color_tvecs;
		cv::Mat left_camera_mat = cv::Mat::eye(3, 3, CV_64F);
		cv::Mat left_dist_coeff = cv::Mat::zeros(8, 1, CV_64F);
		cv::Mat right_camera_mat = cv::Mat::eye(3, 3, CV_64F);
		cv::Mat right_dist_coeff = cv::Mat::zeros(8, 1, CV_64F);
		cv::Mat color_camera_mat = cv::Mat::eye(3, 3, CV_64F);
		cv::Mat color_dist_coeff = cv::Mat::zeros(8, 1, CV_64F);
		bool use_calibrateCameraRO = false;
		double left_rms, right_rms, color_rms;
		if (use_calibrateCameraRO)
		{
			// more accurate
			int i_fixed_pt = board_size.width - 1;
			left_rms = cv::calibrateCameraRO(
				obj_pts, left_corner_pts, img_size, i_fixed_pt,
				left_camera_mat, left_dist_coeff,
				left_rvecs, left_tvecs, left_new_obj_pts,
				0, term_criteia
			);
			right_rms = cv::calibrateCameraRO(
				obj_pts, right_corner_pts, img_size, i_fixed_pt,
				right_camera_mat, right_dist_coeff,
				right_rvecs, right_tvecs, right_new_obj_pts,
				0, term_criteia
			);
			color_rms = cv::calibrateCameraRO(
				obj_pts, color_corner_pts, img_size, i_fixed_pt,
				color_camera_mat, color_dist_coeff, 
				color_rvecs, color_tvecs, color_new_obj_pts,
				0, term_criteia
			);
		}
		else
		{
			left_rms = cv::calibrateCamera(
				obj_pts, left_corner_pts, img_size,
				left_camera_mat, left_dist_coeff, left_rvecs, left_tvecs,
				0, term_criteia
			);
			right_rms = cv::calibrateCamera(
				obj_pts, right_corner_pts, img_size,
				right_camera_mat, right_dist_coeff, right_rvecs, right_tvecs,
				0, term_criteia
			);
			color_rms = cv::calibrateCamera(
				obj_pts, color_corner_pts, img_size,
				color_camera_mat, color_dist_coeff, color_rvecs, color_tvecs,
				0, term_criteia
			);
		}
		std::cout << "RMS error reported by calibrateCamera Left : " << left_rms << std::endl;
		std::cout << "RMS error reported by calibrateCamera Right : " << right_rms << std::endl;
		std::cout << "RMS error reported by calibrateCamera Color : " << right_rms << std::endl;

		if (!(cv::checkRange(left_camera_mat) && cv::checkRange(left_dist_coeff) && cv::checkRange(right_camera_mat) && cv::checkRange(right_dist_coeff) && cv::checkRange(color_camera_mat) && cv::checkRange(color_dist_coeff)))
		{
			std::cout << "Calibration failed" << std::endl;
			return false;
		}

		// ------------------------------- Calibrate Stereo -------------------------------------------
		cv::Mat lr_R, lr_T, lr_E, lr_F;
		double lr_rms = cv::stereoCalibrate(
			obj_pts, left_corner_pts, right_corner_pts,
			left_camera_mat, left_dist_coeff,
			right_camera_mat, right_dist_coeff,
			img_size,
			lr_R, lr_T, lr_E, lr_F,
			cv::CALIB_USE_INTRINSIC_GUESS,
			term_criteia
		);

		std::cout << "Stereo Calibrate Left-Right RMS error : " << lr_rms << std::endl;

		cv::Mat lc_R, lc_T, lc_E, lc_F;
		double lc_rms = cv::stereoCalibrate(
			obj_pts, left_corner_pts, color_corner_pts,
			left_camera_mat, left_dist_coeff,
			color_camera_mat, color_dist_coeff,
			img_size,
			lc_R, lc_T, lc_E, lc_F,
			cv::CALIB_USE_INTRINSIC_GUESS,
			term_criteia
		);

		std::cout << "Stereo Calibrate Left-Color RMS error : " << lc_rms << std::endl;

		// -------------------------------- Stereo Rectify ----------------------------------
		cv::Mat left_R, right_R, left_P, right_P, color_R, color_P, Q;
		cv::Rect valid_roi[2];
		double ratio = cv::rectify3Collinear(
			left_camera_mat, left_dist_coeff,
			right_camera_mat, right_dist_coeff,
			color_camera_mat, color_dist_coeff,
			left_corner_pts, color_corner_pts,
			img_size,
			lr_R, lr_T, lc_R, lc_T,
			left_R, right_R, color_R, left_P, right_P, color_P, Q, -1.,
			img_size, &valid_roi[0], &valid_roi[1], cv::CALIB_ZERO_DISPARITY
		);

		/*
		// --------------------- Caliculate avg reprojection error ----------------------
		obj_pts.clear();
		obj_pts.resize(left_corner_pts.size(), left_new_obj_pts);
		double total_err = 0, err;
		int total_pts = 0;
		std::vector<float> reproj_errs;
		std::vector<cv::Point2f> tmp_img_pts;
		reproj_errs.resize(obj_pts.size());
		for (int i = 0; i < (int)obj_pts.size(); i++)
		{
		cv::projectPoints(cv::Mat(obj_pts[i]), rvecs[i], tvecs[i], camera_mat, dist_coeff, tmp_img_pts);
		err = cv::norm(cv::Mat(corner_pts[i]), cv::Mat(tmp_img_pts), cv::NORM_L2);
		int n = (int)obj_pts[i].size();
		reproj_errs[i] = (float)std::sqrt(err*err / n);
		total_err += err*err;
		total_pts += n;
		}
		double total_avg_err = std::sqrt(total_err / total_pts);
		printf("Calibration succeeded, avg reprojection error = %.7f\n", total_avg_err);
		*/

		// --------------------- Show undistorted images ------------------------------
		cv::Mat left_map1, left_map2, right_map1, right_map2, color_map1, color_map2;
		cv::initUndistortRectifyMap(
			left_camera_mat, left_dist_coeff, left_R, left_P,
			img_size, CV_32FC1, left_map1, left_map2
		);
		cv::initUndistortRectifyMap(
			right_camera_mat, right_dist_coeff, right_R, right_P,
			img_size, CV_32FC1, right_map1, right_map2
		);
		cv::initUndistortRectifyMap(
			color_camera_mat, color_dist_coeff, color_R, color_P,
			img_size, CV_32FC1, color_map1, color_map2
		);

		// ----------------------------- Save as yml file -----------------------------------
		time_t now = time(nullptr);
		struct tm pnow;
		localtime_s(&pnow, &now);
		char date[50];
		sprintf_s(date, "%02d%02d%02d%02d%02d", pnow.tm_mon + 1,
			pnow.tm_mday, pnow.tm_hour, pnow.tm_min, pnow.tm_sec);

		std::string filename = "trinocular_stereo_calibration_" + std::string(date) + ".yml";
		cv::FileStorage fs(filename, cv::FileStorage::WRITE);
		if (!fs.isOpened())
		{
			std::cout << "File can not be opened. " << std::endl;
			return -1;
		}
		fs << "camera_type" << "D435";
		fs << "calibration_time" << date;
		fs << "image_width" << img_size.width;
		fs << "image_height" << img_size.height;
		fs << "board_width" << board_size.width;
		fs << "board_height" << board_size.height;
		fs << "cell_size" << chess_size;
		fs << "left_camera_matrix" << left_camera_mat;
		fs << "left_distortion_error" << left_dist_coeff;
		fs << "right_camera_matrix" << right_camera_mat;
		fs << "right_distortion_error" << right_dist_coeff;
		fs << "color_camera_matrix" << color_camera_mat;
		fs << "color_distortion_error" << color_dist_coeff;
		fs << "left_rms" << left_rms;
		fs << "right_rms" << right_rms;
		fs << "color_rms" << color_rms;
		// fs << "rms" << rms;
		// fs << "avg_reprojection_error" << total_avg_err;
		// fs << "per_view_reprojection_errors" << cv::Mat(reproj_errs);
		fs << "left_R" << left_R;
		fs << "right_R" << right_R;
		fs << "color_R" << color_R;
		fs << "left_P" << left_P;
		fs << "right_P" << right_P;
		fs << "color_P" << color_P;
		fs << "disparity_ratio" << ratio;
		fs << "Q" << Q;
		fs << "left_roi" << valid_roi[0];
		fs << "right_roi" << valid_roi[1];
		fs << "lr_R" << lr_R;
		fs << "lr_T" << lr_T;
		fs << "lr_E" << lr_E;
		fs << "lr_F" << lr_F;
		fs << "lr_rms" << lr_rms;
		fs << "lc_R" << lc_R;
		fs << "lc_T" << lc_T;
		fs << "lc_E" << lc_E;
		fs << "lc_F" << lc_F;
		fs << "lc_rms" << lc_rms;
		fs << "left_map1" << left_map1;
		fs << "left_map2" << left_map2;
		fs << "right_map1" << right_map1;
		fs << "right_map2" << right_map2;
		fs << "color_map1" << color_map1;
		fs << "color_map2" << color_map2;
		fs.release();

		while (true)
		{
			rs2::frameset frameset;
			if (!pipeline.poll_for_frames(&frameset)) continue;
			rs2::frame left_frame = frameset.get_infrared_frame(LEFT);
			rs2::frame right_frame = frameset.get_infrared_frame(RIGHT);
			rs2::frame color_frame = frameset.get_color_frame();
			cv::Mat left_mat(img_size, CV_8UC1, (void*)left_frame.get_data(), cv::Mat::AUTO_STEP);
			cv::Mat right_mat(img_size, CV_8UC1, (void*)right_frame.get_data(), cv::Mat::AUTO_STEP);
			cv::Mat color_mat(img_size, CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

			cv::Mat left_undistorted_mat, right_undistorted_mat, color_undistorted_mat;
			cv::remap(left_mat, left_undistorted_mat, left_map1, left_map2, cv::INTER_LINEAR);
			cv::remap(right_mat, right_undistorted_mat, right_map1, right_map2, cv::INTER_LINEAR);
			cv::remap(color_mat, color_undistorted_mat, color_map1, color_map2, cv::INTER_LINEAR);
			cv::Mat show_undistorted, show_left_undistorted, show_right_undistorted;
			cv::cvtColor(left_undistorted_mat, show_left_undistorted, cv::COLOR_GRAY2BGR);
			cv::cvtColor(right_undistorted_mat, show_right_undistorted, cv::COLOR_GRAY2BGR);

			cv::hconcat(std::vector<cv::Mat>{show_left_undistorted, show_right_undistorted, color_undistorted_mat}, show_undistorted);
			cv::imshow("undistortion", show_undistorted);
			int kk = cv::waitKey(1);
			if (kk == ESC_KEY) break;
		}
		return true;
	}
	else
	{
		std::cout << "number of images is not enough" << std::endl;
		return false;
	}
}

int main() try
{
	const cv::Size board_size(9, 6);
	const float chess_size = 22.f;  // mm

    // --------------------------- RealSense Settings -----------------------------------
	const int width = 848, height = 480, fps = 60;
	const cv::Size img_size(width, height);
	rs2::pipeline pipeline;
	rs2::config rs_cfg;
	rs_cfg.disable_all_streams();
	rs_cfg.enable_stream(RS2_STREAM_INFRARED, LEFT, width, height, RS2_FORMAT_Y8, fps);
	rs_cfg.enable_stream(RS2_STREAM_INFRARED, RIGHT, width, height, RS2_FORMAT_Y8, fps);
	rs_cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
	rs2::pipeline_profile pipeline_profile = pipeline.start(rs_cfg);
	auto depth_sensor = pipeline_profile.get_device().first<rs2::depth_sensor>();
	depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f);

	// -------------------------- Start Calibration ----------------------------------
	trinocular_stereo_calibration(pipeline, img_size, board_size, chess_size);

	return EXIT_SUCCESS;
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