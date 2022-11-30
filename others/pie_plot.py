from turtle import color
import numpy as np
import matplotlib.pyplot as plt


# "0~10%" , "10~20%"  , "20~30%" , "30~40%" , "40~50%"  , "50~60%" , "others"
def pie_chart():
    labels = "0~10%" , "10~20%"  , "20~30%" 
    size = [89.4, 6.1, 4.5]
    # ax1 = plt.subplot2grid((1,2),(0,0))
    plt.figure()
    plt.title('Estrus data : 66 | average : 3.15%', pad='20.0')
    plt.pie(size,labels = labels,autopct="%1.1f%%") 
    plt.axis("equal")

    labels = "0~10%" , "others"
    size = [99.8,0.2]
    # ax1 = plt.subplot2grid((1,2),(0,1))
    plt.figure()
    plt.title('Normal data : 11484 | average : 0.25%', pad='10.0')
    plt.pie(size,labels = labels,autopct="%1.1f%%") 
    plt.axis("equal")

    # size = [1052 , 26 ]
    # ax1 = plt.subplot2grid((2,2),(1,0))
    # plt.title('Dataset3 : 26/1052')
    # plt.pie(size,labels = labels,autopct="%1.1f%%") 
    # plt.axis("equal")

    # size = [90460 , 257 ]
    # ax1 = plt.subplot2grid((2,2),(1,1))
    # plt.title('Dataset4 : 257/90460')
    # plt.pie(size,labels = labels,autopct="%1.1f%%") 
    # plt.axis("equal")
    plt.show()

pie_chart()