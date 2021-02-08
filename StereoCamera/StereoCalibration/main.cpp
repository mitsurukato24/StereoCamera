#include <iostream>
#include <opencv2//opencv.hpp>
#include <librealsense2\rs.hpp>
#include <chrono>

#define ESC_KEY 27
#define SPACE_KEY 32

enum Infrared { LEFT = 1, RIGHT = 2 };

bool calibration(
	rs2::pipeline &pipeline, const Infrared infrared, 
	const cv::Size img_size, const cv::Size board_size, const float chess_size
)
{
	std::chrono::system_clock::time_point start, end;
	int min_num_frames = 10;
	std::vector<std::vector<cv::Point2f>> corner_pts;  // corner points in image coordinates
	while (true)
	{
		start = std::chrono::system_clock::now();

		// ------------------------------------- Get Frame -----------------------------------------------
		rs2::frameset frameset;
		if (!pipeline.poll_for_frames(&frameset)) continue;

		rs2::frame rs2_frame = frameset.get_infrared_frame(infrared);
		cv::Mat frame(img_size, CV_8UC1, (void*)rs2_frame.get_data(), cv::Mat::AUTO_STEP);

		std::vector<cv::Point2f> buf_corner_pts;
		int chess_board_flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
		bool found = cv::findChessboardCorners(frame, board_size, buf_corner_pts, chess_board_flags);

		if (found)
		{
			// Use CornerSubPix to improve accuracy
			cv::Size win_size(11, 11);  // half of search window
			cv::TermCriteria termCriteia(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 50, 0.0001);
			cv::cornerSubPix(frame, buf_corner_pts, win_size, cv::Size(-1, -1), termCriteia);

			cv::Mat show_calib;
			cv::cvtColor(frame, show_calib, cv::COLOR_GRAY2BGR);
			cv::drawChessboardCorners(show_calib, board_size, cv::Mat(buf_corner_pts), found);
			cv::imshow("calibration", frame);

			end = std::chrono::system_clock::now();
			double fps = 1000000.0 / (static_cast<double> (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()));
			// printf("%lf fps\n", fps);

			int key = cv::waitKey(1);
			if (key == ESC_KEY) break;
			else if (key == SPACE_KEY)
			{
				cv::imshow("show picked image", frame);
				int k = cv::waitKey(0);
				if (k == SPACE_KEY) corner_pts.push_back(buf_corner_pts);
			}
		}
		else
		{
			end = std::chrono::system_clock::now();
			double fps = 1000000.0 / (static_cast<double> (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()));
			// printf("%lf fps\n", fps);

			cv::imshow("calibration", frame);
			int key = cv::waitKey(1);
			if (key == ESC_KEY) break;
		}
	}

	// --- calculate intrinsic and extrinsic parameters
	if ((int)corner_pts.size() >= min_num_frames)
	{
		std::vector<cv::Point3f> new_obj_pts;
		std::vector<std::vector<cv::Point3f>> obj_pts(1);  // true 3d coordinates of corners
		for (int h = 0; h < board_size.height; h++)
		{
			for (int w = 0; w < board_size.width; w++)
			{
				obj_pts[0].push_back(cv::Point3f(chess_size*w, chess_size*h, 0));
			}
		}
		new_obj_pts = obj_pts[0];
		obj_pts.resize(corner_pts.size(), obj_pts[0]); // copy
		std::vector<cv::Mat> rvecs, tvecs;
		cv::Mat camera_mat = cv::Mat::eye(3, 3, CV_64F);
		cv::Mat dist_coeff = cv::Mat::zeros(8, 1, CV_64F);
		bool use_calibrateCameraRO = false;
		double rms;
		if (use_calibrateCameraRO)
		{
			// more accurate
			int iFixedPt = board_size.width - 1;
			rms = cv::calibrateCameraRO(
				obj_pts, corner_pts, img_size, iFixedPt, camera_mat, dist_coeff,
				rvecs, tvecs, new_obj_pts
			);
		}
		else
		{
			rms = cv::calibrateCamera(
				obj_pts, corner_pts, img_size, camera_mat, dist_coeff, rvecs, tvecs
			);
		}
		std::cout << "RMS error reported by calibrateCamera: " << rms << std::endl;

		if (!(cv::checkRange(camera_mat) && cv::checkRange(dist_coeff)))
		{
			std::cout << "Calibration failed" << std::endl;
			return false;
		}

		// --- caliculate avg reprojection error
		obj_pts.clear();
		obj_pts.resize(corner_pts.size(), new_obj_pts);
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

		// --- save as yml file
		time_t now = time(nullptr);
		struct tm pnow;
		localtime_s(&pnow, &now);
		char date[50];
		sprintf_s(date, "%02d%02d%02d%02d%02d", pnow.tm_mon + 1,
			pnow.tm_mday, pnow.tm_hour, pnow.tm_min, pnow.tm_sec);

		std::string left_or_right = infrared == LEFT ? "left" : "right";
		std::string filename = left_or_right + "_calibration_" + std::string(date) + ".yml";
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
		fs << "camera_matrix" << camera_mat;
		fs << "tvecs" << tvecs;
		fs << "rvecs" << rvecs;
		fs << "distortion_error" << dist_coeff;
		fs << "avg_reprojection_error" << total_avg_err;
		fs << "per_view_reprojection_errors" << cv::Mat(reproj_errs);

		fs.release();

		// --- show undistorted images
		cv::Mat map1, map2;
		cv::initUndistortRectifyMap(camera_mat, dist_coeff, cv::Mat(),
			cv::getOptimalNewCameraMatrix(camera_mat, dist_coeff, img_size, 1, img_size, 0),
			img_size, CV_16SC2, map1, map2);

		cv::Mat frame_undistorted;
		while (true)
		{
			rs2::frameset frameset;
			if (!pipeline.poll_for_frames(&frameset)) continue;
			rs2::frame rs2_frame = frameset.get_infrared_frame(infrared);
			cv::Mat frame(img_size, CV_8UC1, (void*)rs2_frame.get_data(), cv::Mat::AUTO_STEP);

			cv::remap(frame, frame_undistorted, map1, map2, cv::INTER_LINEAR);
			cv::resize(frame, frame, frame_undistorted.size());
			cv::hconcat(std::vector<cv::Mat>{ frame, frame_undistorted }, frame);
			cv::imshow("undistortion", frame);
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

bool stereo_calibration(
	rs2::pipeline &pipeline, 
	const cv::Size img_size, const cv::Size board_size, const float chess_size
)
{
	std::chrono::system_clock::time_point start, end;
	int min_num_frames = 10;
	std::vector<std::vector<cv::Point2f>> left_corner_pts, right_corner_pts;  // corner points in image coordinates
	cv::TermCriteria term_criteia(
		cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
		100,  // max number of iterations
		0.00001  // min accuracy
	);
	while (true)
	{
		start = std::chrono::system_clock::now();

		// -------------------------------- Get Frame --------------------------------------
		rs2::frameset frameset;
		if (!pipeline.poll_for_frames(&frameset)) continue;

		rs2::frame left_frame = frameset.get_infrared_frame(LEFT);
		rs2::frame right_frame = frameset.get_infrared_frame(RIGHT);
		cv::Mat left_mat(img_size, CV_8UC1, (void*)left_frame.get_data(), cv::Mat::AUTO_STEP); 
		cv::Mat right_mat(img_size, CV_8UC1, (void*)right_frame.get_data(), cv::Mat::AUTO_STEP);

		std::vector<cv::Point2f> left_buf_corner_pts, right_buf_corner_pts;
		int chess_board_flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
		bool left_found = cv::findChessboardCorners(left_mat, board_size, left_buf_corner_pts, chess_board_flags);
		bool right_found = cv::findChessboardCorners(right_mat, board_size, right_buf_corner_pts, chess_board_flags);

		if (left_found && right_found)
		{
			// -------------------- Use CornerSubPix to improve accuracy ------------------
			cv::Size win_size(11, 11);  // half of search window
			cv::cornerSubPix(left_mat, left_buf_corner_pts, win_size, cv::Size(-1, -1), term_criteia);
			cv::cornerSubPix(right_mat, right_buf_corner_pts, win_size, cv::Size(-1, -1), term_criteia);

			end = std::chrono::system_clock::now();
			double fps = 1000000.0 / (static_cast<double> (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()));
			// printf("%lf fps\n", fps);

			// --------------------------- Draw chessboard ----------------------------------
			cv::Mat show_left, show_right;
			cv::cvtColor(left_mat, show_left, cv::COLOR_GRAY2BGR);
			cv::cvtColor(right_mat, show_right, cv::COLOR_GRAY2BGR);
			cv::drawChessboardCorners(show_left, board_size, cv::Mat(left_buf_corner_pts), left_found);
			cv::drawChessboardCorners(show_right, board_size, cv::Mat(right_buf_corner_pts), right_found);
			cv::Mat show_calib;
			cv::hconcat(show_left, show_right, show_calib);
			cv::imshow("calibration", show_calib);
			int key = cv::waitKey(1);
			if (key == ESC_KEY) break;
			else if (key == SPACE_KEY)
			{
				cv::imshow("show picked image", show_calib);
				int k = cv::waitKey(0);
				if (k == SPACE_KEY)
				{
					left_corner_pts.push_back(left_buf_corner_pts);
					right_corner_pts.push_back(right_buf_corner_pts);
				}
			}
		}
		else
		{
			end = std::chrono::system_clock::now();
			double fps = 1000000.0 / (static_cast<double> (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()));
			// printf("%lf fps\n", fps);

			cv::Mat show_img;
			cv::hconcat(left_mat, right_mat, show_img);
			cv::imshow("not found", show_img);
			int key = cv::waitKey(1);
			if (key == ESC_KEY) break;
		}
	}

	// --------------- Calculate intrinsic and extrinsic parameters -------------------------
	if ((int)left_corner_pts.size() >= min_num_frames)
	{
		std::vector<cv::Point3f> left_new_obj_pts, right_new_obj_pts;
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
		obj_pts.resize(left_corner_pts.size(), obj_pts[0]); // copy

		std::vector<cv::Mat> left_rvecs, left_tvecs, right_rvecs, right_tvecs;
		cv::Mat left_camera_mat = cv::Mat::eye(3, 3, CV_64F);
		cv::Mat right_camera_mat = cv::Mat::eye(3, 3, CV_64F);
		cv::Mat left_dist_coeff = cv::Mat::zeros(8, 1, CV_64F);
		cv::Mat right_dist_coeff = cv::Mat::zeros(8, 1, CV_64F);
		bool use_calibrateCameraRO = false;
		double left_rms, right_rms;
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
		}
		else
		{
			left_rms = cv::calibrateCamera(
				obj_pts, left_corner_pts, img_size,
				left_camera_mat, left_dist_coeff, left_rvecs, left_tvecs,
				0, term_criteia
			);
			right_rms = cv::calibrateCamera(
				obj_pts, left_corner_pts, img_size,
				right_camera_mat, right_dist_coeff, right_rvecs, right_tvecs,
				0, term_criteia
			);
		}
		std::cout << "RMS error reported by calibrateCamera Left :" << left_rms << std::endl;
		std::cout << "RMS error reported by calibrateCamera Right :" << right_rms << std::endl;

		if (!(cv::checkRange(left_camera_mat) && cv::checkRange(left_dist_coeff) && cv::checkRange(right_camera_mat) && cv::checkRange(right_dist_coeff)))
		{
			std::cout << "Calibration failed" << std::endl;
			return false;
		}

		cv::Mat R, T, E, F;
		double rms = cv::stereoCalibrate(
			obj_pts, left_corner_pts, right_corner_pts,
			left_camera_mat, left_dist_coeff,
			right_camera_mat, right_dist_coeff, 
			img_size,
			R, T, E, F, 
			cv::CALIB_FIX_INTRINSIC,
			term_criteia
		);

		std::cout << "Stereo Calibrate RMS error : " << rms << std::endl;
		
		// -------------------------------- Stereo Rectify ----------------------------------
		cv::Mat left_R, right_R, left_P, right_P, Q;
		cv::Rect valid_roi[2];
		cv::stereoRectify(
			left_camera_mat, left_dist_coeff,
			right_camera_mat, right_dist_coeff,
			img_size, R, T,
			left_R, right_R, left_P, right_P, Q,
			cv::CALIB_ZERO_DISPARITY, 1,
			img_size, &valid_roi[0], &valid_roi[1]
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
		cv::Mat left_map1, left_map2, right_map1, right_map2;
		cv::initUndistortRectifyMap(
			left_camera_mat, left_dist_coeff, left_R, left_P,
			img_size, CV_32FC1, left_map1, left_map2
		);
		cv::initUndistortRectifyMap(
			right_camera_mat, right_dist_coeff, right_R, right_P,
			img_size, CV_32FC1, right_map1, right_map2
		);

		// ----------------------------- Save as yml file -----------------------------------
		time_t now = time(nullptr);
		struct tm pnow;
		localtime_s(&pnow, &now);
		char date[50];
		sprintf_s(date, "%02d%02d%02d%02d%02d", pnow.tm_mon + 1,
			pnow.tm_mday, pnow.tm_hour, pnow.tm_min, pnow.tm_sec);

		std::string filename = "stereo_calibration_" + std::string(date) + ".yml";
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
		fs << "rms" << rms;
		// fs << "avg_reprojection_error" << total_avg_err;
		// fs << "per_view_reprojection_errors" << cv::Mat(reproj_errs);
		fs << "left_R" << left_R;
		fs << "right_R" << right_R;
		fs << "left_P" << left_P;
		fs << "right_P" << right_P;
		fs << "Q" << Q;
		fs << "left_roi" << valid_roi[0];
		fs << "right_roi" << valid_roi[1];
		fs << "R" << R;
		fs << "T" << T;
		fs	<< "E" << E;
		fs << "F" << F;
		fs << "left_map1" << left_map1;
		fs << "left_map2" << left_map2;
		fs << "right_map1" << right_map1;
		fs << "right_map2" << right_map2;
		fs.release();
		
		while (true)
		{
			rs2::frameset frameset;
			if (!pipeline.poll_for_frames(&frameset)) continue;
			rs2::frame left_frame = frameset.get_infrared_frame(LEFT);
			rs2::frame right_frame = frameset.get_infrared_frame(RIGHT);
			cv::Mat left_mat(img_size, CV_8UC1, (void*)left_frame.get_data(), cv::Mat::AUTO_STEP);
			cv::Mat right_mat(img_size, CV_8UC1, (void*)right_frame.get_data(), cv::Mat::AUTO_STEP);

			cv::Mat left_undistorted_mat, right_undistorted_mat;
			cv::remap(left_mat, left_undistorted_mat, left_map1, left_map2, cv::INTER_LINEAR);
			cv::remap(right_mat, right_undistorted_mat, right_map1, right_map2, cv::INTER_LINEAR);
			cv::Mat show_undistorted;
			cv::hconcat(left_undistorted_mat, right_undistorted_mat, show_undistorted);
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
	const int width = 848, height = 480, fps = 90;
	const cv::Size img_size(width, height);
	rs2::pipeline pipeline;
	rs2::config rs_cfg;
	rs_cfg.disable_all_streams();
	rs_cfg.enable_stream(RS2_STREAM_INFRARED, LEFT, width, height, RS2_FORMAT_Y8, fps);
	rs_cfg.enable_stream(RS2_STREAM_INFRARED, RIGHT, width, height, RS2_FORMAT_Y8, fps);
	rs2::pipeline_profile pipeline_profile = pipeline.start(rs_cfg);
	auto depth_sensor = pipeline_profile.get_device().first<rs2::depth_sensor>();
	depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f);

	// calibration(pipeline, LEFT, img_size, board_size, chess_size);
	// calibration(pipeline, RIGHT, img_size, board_size, chess_size);

	stereo_calibration(pipeline, img_size, board_size, chess_size);

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