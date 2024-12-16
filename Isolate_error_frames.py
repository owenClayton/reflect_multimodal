import cv2
import json
import os 
from datetime import datetime, timedelta

video_folder = ''
reasoning_folder = ''
# call on thor_tasks folder for video and state_summary folder for the frames to isolate. Isolate 4 frames (one before predicted, one at predicted(might be multiple), and one after predicted). See what is produced.
# save these frames in the state_summary folder. 

simPath = '/pub0/oclayton/reflect_multimodal/main/failed_tasks/state_summary'
for root1, dirs1, files1 in os.walk(simPath):
    for dir1 in dirs1:
        pathToTasks = os.path.join(root1, dir1)
        with open(f'{pathToTasks}/reasoning.json') as f:
            reasoning = json.load(f)
            failure_times = reasoning["pred_failure_step"]
            frame_snip_times = []
            for time in failure_times:
                time_obj = datetime.strptime(time, "%M:%S")
                frame_snip_times.append(time_obj.minute*60 + time_obj.second  -3)
                frame_snip_times.append(time_obj.minute*60 + time_obj.second)
                frame_snip_times.append(time_obj.minute*60 + time_obj.second + 3)
            video = cv2.VideoCapture(f'/pub0/oclayton/reflect_multimodal/main/failed_tasks/thor_tasks/{dir1}/original-video.mp4')
            for frame_time in frame_snip_times: 
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_time)
                ret, frame = video.read()
                if(ret):
                    cv2.imwrite(f'{pathToTasks}/{frame_time}.png', frame)
                
    break

