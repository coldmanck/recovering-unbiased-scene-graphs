# To be used insitde `output/` folder!
import os

root_folder = 'output'

for folder in os.listdir(root_folder):
    cur_path = os.path.join(root_folder, folder)
    if os.path.isdir(cur_path):
        print('--------', cur_path, '--------')
        model_paths = sorted([os.path.join(cur_path, f) for f in os.listdir(cur_path) if f.startswith('model_')])
        if len(model_paths) < 2:
            print('Less than 2 models. Skip.')
            continue
        to_delete_paths = model_paths[:-1]
        newest_model_path = model_paths[-1]
        
        print('The newest model:', newest_model_path)
        print('To delete models:')
        for model_path in to_delete_paths:
            print(model_path)
            
print('Confirm to delete files above? (y/n)')
if input() == 'y':
    for folder in os.listdir(root_folder):
        cur_path = os.path.join(root_folder, folder)
        if os.path.isdir(cur_path):
            print('--------', cur_path, '--------')
            model_paths = sorted([os.path.join(cur_path, f) for f in os.listdir(cur_path) if f.startswith('model_')])
            if len(model_paths) < 2:
                print('Less than 2 models. Skip.')
                continue
            to_delete_paths = model_paths[:-1]
            newest_model_path = model_paths[-1]
            
            print('The newest model:', newest_model_path)
            print('To delete models:')
            for model_path in to_delete_paths:
                os.remove(model_path)
                print(f'{model_path} deleted.')
else:
    print('Leave without deleting any file.')
    