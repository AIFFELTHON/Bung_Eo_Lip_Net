import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects



import os

def show_video_subtitle(frames, subtitle):
    fig, ax = plt.subplots()    
    fig.show()

    text = plt.text(0.5, 0.1, "", 
        ha='center', va='center', transform=ax.transAxes, 
        fontdict={'fontsize': 15, 'color':'white', 'fontweight': 500})
    # plt.text로 text 생성 인수의 자세한 설명은 plt document를 참고하는 것이 더 좋을 것 같다.
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
        path_effects.Normal()])
    # set_path_effects의 경로 지정, Stroke -> 획을 다시 그어준다는데 뭔소린지 모르겠음.
    subs = subtitle.split() 
    inc = max(len(frames)/(len(subs)+1), 0.01)

    img = None
    for i, frame in enumerate(frames):
        sub = " ".join(subs[:int(i/inc)]) #split한 subtitle[:int(i/inc)]까지 join

        text.set_text(sub)  #plt.text plt.text.set_text

        if img is None:
            img = plt.imshow(frame)
        else:
            img.set_data(frame)

        i = f'{i}'.zfill(2)
        save_path = os.path.join(os.getcwd(), f'results/result_{i}.png')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        plt.savefig(filename=save_path, bbox_inches='tight', pad_inches=0)
        
        fig.canvas.draw()