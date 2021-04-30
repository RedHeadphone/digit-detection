import pygame as p
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

side=588
numbox=28
eachlen=side//numbox

p.init()
screen=p.display.set_mode((side,side+60))
p.display.set_caption("Path Finding")

model = tf.keras.models.load_model('digits.model')

class Button:
    def __init__(self,x,y,st,c1,c2,w):
        self.x=x
        self.y=y
        self.bool=True
        self.st=st
        self.color1=c1
        self.color2=c2
        self.width=w
        self.lastclick=0

    def check(self,x1,y1,click):
        global blocks
        if self.x<x1<self.x+100 and self.y<y1<self.y+40:
            self.bool=False
            if click==0 and self.lastclick==1:
                if self.st=="reset":
                    blocks=[[0 for i in range(numbox)] for i in range(numbox)]
                if self.st=="detect digit":
                    prediction = model.predict([blocks])
                    print(np.argmax(prediction))
            self.lastclick=click
        else :
            self.bool=True
    def draw_button(self):
        if self.bool:
            c1=self.color1
            c2=self.color2
        else :
            c2=self.color1
            c1=self.color2
        font = p.font.Font('freesansbold.ttf',18)
        text = font.render(self.st, True, c1) 
        textRect = text.get_rect()
        textRect.center = (self.x+self.width//2, self.y+20)
        p.draw.rect(screen,c2,[(self.x,self.y),(self.width,40)])
        screen.blit(text, textRect)

blocks=[[0 for i in range(numbox)] for i in range(numbox)]

def draw_grid():
    for i in range(numbox):
        for j in range(numbox):
            if blocks[j][i]==1:
                p.draw.rect(screen,(0,0,0),[(i*eachlen,j*eachlen),(eachlen,eachlen)])
    for x in range(1,numbox+1):
        p.draw.line(screen,(0,0,0),(0,x*eachlen),(side,x*eachlen),2)
    for y in range(1,numbox+1):
        p.draw.line(screen,(0,0,0),(y*eachlen,0),(y*eachlen,side),2)

done=False
Bs=[Button(10,598,"reset",(75,194,197),(52,132,152),100),Button(428,598,"detect digit",(200,200,30),(200,100,30),150)]

while not done:
    for event in p.event.get():
        if event.type==p.QUIT:
            done=True
    mo=p.mouse.get_pressed()
    mot0,mot1=p.mouse.get_pos()
    if mo[0]==1 and mot1<side:
        pos=((mot0//eachlen),(mot1//eachlen))
        blocks[pos[1]][pos[0]]=1

    screen.fill((255,255,255))
    draw_grid()
    for b in Bs:
        b.check(mot0,mot1,mo[0])
        b.draw_button()
    p.display.update()
p.quit()