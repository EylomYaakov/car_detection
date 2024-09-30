#include <pangolin/utils/file_utils.h>
#include <pangolin/geometry/glgeometry.h>

#include "Auxiliary.h"
#include <simulator/simulator.h>
#include "RunModel/TextureShader.h"
#include <cmath>
#include <vector>
#include <string>
#include <pybind11/embed.h>
#include <fstream>

namespace py = pybind11;


void check_traffic();
void move_simulator(Simulator& simulator, std::vector<Eigen::Vector3d> targets, 
std::vector<Eigen::Vector3d> colors, std::vector<std::string> videos, std::vector<int> road_parameters);



int main(int argc, char **argv) {
    //initialize the simulator
    std::string settingPath = Auxiliary::GetGeneralSettingsPath();
    std::ifstream programData(settingPath);
    nlohmann::json data;
    programData >> data;
    programData.close();
    std::string configPath = data["slam_configuration"]["drone_yaml_path"];
    std::string vocabulary_path = data["slam_configuration"]["vocabulary_path"];
    std::string modelTextureNameToAlignTo = data["simulator_configuration"]["align_model_to_texture"]["texture"];
    bool alignModelToTexture = data["simulator_configuration"]["align_model_to_texture"]["align"];
    std::string model_path = data["simulator_configuration"]["model_path"];
    std::string simulatorOutputDir = data["simulator_configuration"]["simulator_output_dir"];
    std::string loadMapPath = data["slam_configuration"]["load_map_path"];
    double movementFactor = data["simulator_configuration"]["movement_factor"];
    double simulatorStartingSpeed = data["simulator_configuration"]["simulator_starting_speed"];
    bool runWithSlam = data["simulator_configuration"]["run_with_slam"];
    Eigen::Vector3f startingPoint((float)data["simulator_configuration"]["starting_pose"]["x"], (float)data["simulator_configuration"]["starting_pose"]["y"],
                                (float)data["simulator_configuration"]["starting_pose"]["z"]);

    Simulator simulator(configPath, model_path, alignModelToTexture, modelTextureNameToAlignTo, startingPoint, false, simulatorOutputDir, false,
                        loadMapPath, movementFactor, simulatorStartingSpeed, vocabulary_path);
    auto simulatorThread = simulator.run();

    // wait for the 3D model to load
    while (!simulator.isReady()) {
        usleep(1000);
    }

    // If with slam, wait for slam to load
    if (runWithSlam) {
        //simulator.setTrack(true);
        while (!simulator.startScanning()) {
            usleep(10);
        }
        // Waits here until Tab is clicked!
    }
    
    //create target points(represents the roads)
    Eigen::Vector3d first_target_point(5.5, -1, 3.1);
    Eigen::Vector3d second_target_point(-2.06667, -2.03333, 3.1);
    Eigen::Vector3d third_target_point(0.0569792, -0.483333, -1.07057);
    Eigen::Vector3d fourth_target_point(3.14807, -1.51667, -1.28437);
    Eigen::Matrix4d pose = simulator.getCurrentLocation();
    Eigen::Vector3d fifth_target_point(pose(0,3), pose(1,3), pose(2,3));
    std::vector<Eigen::Vector3d> targets;
    targets.push_back(first_target_point);
    targets.push_back(second_target_point);
    targets.push_back(third_target_point);
    targets.push_back(fourth_target_point);
    targets.push_back(fifth_target_point);
    //create colors(to mark the target points)
    Eigen::Vector3d red(255,0,0);   
    Eigen::Vector3d blue(0,0,255);
    Eigen::Vector3d green(0, 255, 0);
    Eigen::Vector3d yellow(255, 165, 0);
    Eigen::Vector3d white(238, 130, 238);
    std::vector<Eigen::Vector3d> colors;
    colors.push_back(red);
    colors.push_back(blue);
    colors.push_back(green);
    colors.push_back(yellow);
    colors.push_back(white);
    //create vector of the paths of the videos, one video for each road
    std::vector<std::string> videos;
    videos.push_back("/home/eylom/Downloads/videos/regular_traffic.mov");
    videos.push_back("/home/eylom/Downloads/videos/big_traffic_jam.mov");
    videos.push_back("/home/eylom/Downloads/videos/small_traffic_jam.mp4");
    videos.push_back("/home/eylom/Downloads/videos/clean_road.mp4");
    videos.push_back("/home/eylom/Downloads/videos/regular_traffic2.mp4");
    //create vector contains the parameter of each road(the maximum amount of cars can be in the road in the time period of the video)
    std::vector<int> road_parameters = {65, 50, 85, 110, 90}; 
    py::initialize_interpreter(); //initialize python interpreter
    move_simulator(simulator, targets, colors, videos, road_parameters);
    py::finalize_interpreter(); //finalize the interpreter
    simulatorThread.join();
    return 0;
}


void move_simulator(Simulator& simulator, std::vector<Eigen::Vector3d> targets, 
std::vector<Eigen::Vector3d> colors, std::vector<std::string> videos, std::vector<int> road_parameters){
    int i = 0;
    int painted = 0;
    //run this until the user close the program - represents a drone that moves in an infinte cycle between roads and scan for traffic
    while(true){
        Eigen::Vector3d current_target = targets[i];
        double targetPointX = current_target[0];
        double targetPointY = current_target[1];
        double targetPointZ = current_target[2];

        //move the drone to the currnent target point
        Eigen::Matrix4d pose = simulator.getCurrentLocation();

        std::string forward = "forward " + to_string(targetPointZ-pose(2,3));
        simulator.command(forward);
        std::string left = "left " + to_string(targetPointX-pose(0,3));
        simulator.command(left);
        std::string down = "down " + to_string(targetPointY-pose(1,3));
        simulator.command(down);
        Eigen::Matrix4d new_pose = simulator.getCurrentLocation();
        int pointSize = 30;
        //if the point not already painted(first time in the point), draw a squre in the point
        if(painted==0){
            simulator.drawPoint(cv::Point3d(new_pose(0, 3), -new_pose(1, 3), new_pose(2, 3)), pointSize, colors[i]);
        }
        //move a little back(in order to see the painted squre, we cant see the painted squre if we are exactly on the point)
        simulator.command("back 0.5");
        //write the current video path and road parameter to files, so the python code could read them from the file
        std::ofstream path("/home/eylom/Downloads/video_path.txt");
        path << videos[i];
        path.close();
        std::ofstream param("/home/eylom/Downloads/road_parameter.txt");
        param << road_parameters[i];
        param.close();
        //run the python code that checks for traffic
        check_traffic();
        i++;
        //if we are in the end of the points list, go back to the start(the drone is moving in a cycle)
        if(i==targets.size()){
            i = 0;
            if(painted == 0){
                painted++;
            }
        }
    }
}


void check_traffic(){
    std::string pythonCode;
    std::getline(std::ifstream("/home/eylom/Downloads/car_detection.py"), pythonCode, '\0'); //read the python code
    //execute the code
    py::exec(pythonCode);
}

