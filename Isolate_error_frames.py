import cv2
failed_list = ['heatPotato-1', 'cookEgg-8', 'makeCoffee-2', 'switchDevices-8', 'toastBread-1', 'warmWater-9', 'waterPlant-2', 'waterPlant-6', 'makeSalad-5', 'makeSalad-10'] #maybe just make new folder of failed tasks to eval
video_folder = ''
reasoning_folder = ''
# call on thor_tasks folder for video and state_summary folder for the frames to isolate. Isolate 4 frames (one before predicted, one at predicted(might be multiple), and one after predicted). See what is produced.
# save these frames in the state_summary folder. 
video = cv2.VideoCapture('/pub0/oclayton/reflect_multimodal/completed_tasks/boilWater/boilWater-1/original-video.mp4')
fps = video.get(cv2.CAP_PROP_FPS)
print(fps)
video.set(cv2.CAP_PROP_POS_FRAMES, 3)
ret, frame = video.read()
cv2.imwrite('my_video_frame.png', frame)