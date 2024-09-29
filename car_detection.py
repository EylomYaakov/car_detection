import cv2
import torch
from ultralytics import YOLO, solutions

def count_objects(points, video_path):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    classes_to_count = [2, 3, 5, 7]  # all vehicles

    # Video writer
    video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                (w + 800, h + 800))

    # Init Object Counter
    counter = solutions.ObjectCounter(
        view_img=True,
        reg_pts=points,
        names=model.names,
        draw_tracks=True,
        line_thickness=2,
    )

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        tracks = model.track(im0, persist=True, show=False, classes=classes_to_count, verbose=False, device = "cpu") # run on the cpu because its fatster in the vm

        im0 = counter.start_counting(im0, tracks)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


    return counter.in_counts + counter.out_counts

def detect_traffic(video_path, road_max):
    #count the total amount of the cars in the video
    total_cars = count_objects([(1920,1080), (1920,0), (0,0), (0, 1080)], video_path)
    #count the amount of cars the cross a certain line
    line_cars = count_objects([(0,500), (2000, 500)], video_path)
    #divide both of them by the max cars the road can contain(to get the precentage of the road that is full)
    total_cars /= road_max
    line_cars /= road_max
    #first option: a lot of cars crossed the line, but even a lot more are counted in the video, meaning a lot cars are in the video but didnt managed to cross the line, indicates a big traffic jam
    if line_cars <= 0.4 and total_cars - line_cars >= 0.4:
        print("big traffic jam")
    #second option: same, but a little less cars are counted and didnt manage to cross the line, indicates a smaller traffic jam
    elif line_cars <= 0.4 and total_cars - line_cars >= 0.2:
        print("small traffic jam")
    #third option: not a lot of cars crossed the line, but also very little(if any) cars counted and didnt cross the line, indicates that the road is almost clean
    elif line_cars <= 0.4:
        print("the road is clean")
    #fourth option: a lot of cars counted, but almost all of them crossed the line, indicated a regualr and standart traffic
    elif line_cars  >= 0.4 and total_cars - line_cars <= 0.2:
        print("regular traffic")
    #fifth option: a lot of cars crossed the line, but also quite a bit amount of cars counted but didnt managed to cross the line, indicates also about small traffic jam
    else:
        print("small traffic jam")

def main():
    #open the files that the c++ code worte the path and road paramter to them, and read them
    path_file = open("/home/eylom/Downloads/video_path.txt", "r")
    video_path = path_file.read()
    path_file.close()
    parameter_file = open("/home/eylom/Downloads/road_parameter.txt", "r")
    road_max = int(parameter_file.read())
    parameter_file.close()
    detect_traffic(video_path, road_max)
    
if __name__ == "__main__":
    main()
