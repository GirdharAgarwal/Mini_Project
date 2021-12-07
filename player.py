import pygame #used to create video games
import tkinter as tkr #used to develop GUI
from tkinter.filedialog import askdirectory #it permit to select dir
import os #it permits to interact with the operating system

def MusicPlayer(emotion):
    music_player = tkr.Tk() 
    music_player.title(emotion)
    music_player.geometry('700x700')
    directory="C:/Users/GIRDHAR AGARWAL/Desktop/mini_project/music/"+str(emotion)
    print(directory)
    os.chdir(directory) #it permits to chenge the current dir
    song_list = os.listdir() #it returns the list of files song
    play_list = tkr.Listbox(music_player, font='Helvetica 12 bold', bg='yellow', selectmode=tkr.SINGLE)
    for item in song_list:
        pos = 0
        play_list.insert(pos, item)
        pos += 1
    pygame.init()
    pygame.mixer.init()
    def play():
        pygame.mixer.music.load(play_list.get(tkr.ACTIVE))
        var.set(play_list.get(tkr.ACTIVE))
        pygame.mixer.music.play()
    def stop():
        pygame.mixer.music.stop()
    def pause():
        pygame.mixer.music.pause()
    def unpause():
        pygame.mixer.music.unpause()
    Button1 = tkr.Button(music_player, width=5, height=3, font='Helvetica 12 bold', text='PLAY', command=play, bg='blue', fg='white')
    Button2 = tkr.Button(music_player, width=5, height=3, font='Helvetica 12 bold', text='STOP', command=stop, bg='red', fg='white')
    Button3 = tkr.Button(music_player, width=5, height=3, font='Helvetica 12 bold', text='PAUSE', command=pause, bg='purple', fg='white')
    Button4 = tkr.Button(music_player, width=5, height=3, font='Helvetica 12 bold', text='UNPAUSE', command=unpause, bg='orange', fg='white')
    var = tkr.StringVar() 
    song_title = tkr.Label(music_player, font='Helvetica 12 bold', textvariable=var)
    song_title.pack()
    Button1.pack(fill='x')
    Button2.pack(fill='x')
    Button3.pack(fill='x')
    Button4.pack(fill='x')
    play_list.pack(fill='x')
    music_player.mainloop()
