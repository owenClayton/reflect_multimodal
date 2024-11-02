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
import zarr

API_KEY= ''
llm_prompter = LLMPrompter(gpt_version="gemini-1.5-flash", api_key_str=API_KEY)

meta_data = zarr.open('/pub0/oclayton/reflect_dataset/real_data/appleInFridge1/replay_buffer.zarr', 'r') #this is then indexable 
subset = meta_data['data/action']
array = np.array(subset)
print(array)


'''
with open(f'/pub0/oclayton/reflect_multimodal/main/tasks_real_world.json') as f:
    tasks = json.load(f)
for x in range(1,31):
    task_info = tasks[f"Task {x}"]
    FOLDER_NAME = f'{tasks["general_folder_name"]}'
    print(FOLDER_NAME)
    #show_video(f'thor_tasks/{FOLDER_NAME}/original-video.mp4')
    WITH_AUDIO = 1 # 1: using audio deteceted with wav2clip, 0: using ground truth audio information
    events, task, object_list, interact_actions, nav_actions = load_data(f"thor_tasks/{FOLDER_NAME}", task_info)
    print(len(events))

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

'''