import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory of 'main' to the Python path
sys.path.append(os.path.join(current_dir, 'main'))

from IPython.display import HTML
from base64 import b64encode


from main.gen_data import *
from main.data import load_data
from main.exp import *
from main.execute_replan import run_correction
from LLM.prompt import LLMPrompter
from pathlib import Path
from main.constants import *

API_KEY= ''
llm_prompter = LLMPrompter(gpt_version="gemini-1.5-flash", api_key_str=API_KEY)

def show_video(video_path, video_width=300):
  video_file = open(video_path, "r+b").read()
  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"""<video width={video_width} controls><source src="{video_url}"></video>""")

simPath = '/pub0/oclayton/reflect_multimodal/main/thor_tasks/makeCoffee/makeCoffee-2'


with open(f'{simPath}/task.json') as f:
    task_info = json.load(f)
#run_data_gen(data_path=os.getcwd(), task=task_info)
FOLDER_NAME = f'{task_info["specific_folder_name"]}'
print(FOLDER_NAME)
#show_video(f'thor_tasks/{FOLDER_NAME}/original-video.mp4')
WITH_AUDIO = 1 # 1: using audio deteceted with wav2clip, 0: using ground truth audio information
events, task, object_list, interact_actions, nav_actions = load_data(f"thor_tasks/{FOLDER_NAME}", task_info)
print("Events ")
print(events)
print("Task: ")
print(task)
print("OBJECT: ")
print(object_list)



# Sensory-input summary
detected_sounds = []
if WITH_AUDIO == 1:
    detected_sounds = run_sound_module(FOLDER_NAME, object_list)
generate_scene_graphs(FOLDER_NAME, events, object_list, nav_actions, interact_actions, WITH_AUDIO, detected_sounds)
with open(f'state_summary/{FOLDER_NAME}/global_sg.pkl', 'rb') as f:
    global_sg = pickle.load(f)
    print("================ Global SG ================")
    print(global_sg)

# Event-based summary & Subgoal-based summary
generate_summary(FOLDER_NAME, events, nav_actions, interact_actions, WITH_AUDIO, detected_sounds)
run_reasoning(FOLDER_NAME, llm_prompter, global_sg)
generate_replan(FOLDER_NAME, llm_prompter, global_sg, events[-1], object_list)
run_correction(data_path=os.getcwd(), f_name=FOLDER_NAME)  
#show_video(f'recovery/{FOLDER_NAME}/recovery-video.mp4')
