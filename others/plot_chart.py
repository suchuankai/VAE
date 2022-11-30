from turtle import color
import numpy as np
import matplotlib.pyplot as plt



def pie_chart():
    labels = "0~10%" , "10~20%"  , "20~30%" , "30~40%" , "40~50%" , "50~60%" , "60~70%" , "70~80%" , "80~90%" , "90~100%"
    size = [59, 4, 3, 0, 0, 0, 0, 0, 0, 0]
    ax1 = plt.subplot2grid((1,2),(0,0))
    plt.title('Estrus data : 66 | average : 3.15%')
    plt.pie(size,labels = labels,autopct="%1.1f%%") 
    plt.axis("equal")

    size = [11461, 16, 6, 1, 0, 0, 0, 0, 0, 0]
    ax1 = plt.subplot2grid((1,2),(0,1))
    plt.title('Normal data : 11484 | average : 0.25%')
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

def bar_chart_each():
    act = ['In_alleys', 'Rest', 'Eat', 'Activity level']

    normal = [743.413,  1899.779, 950.424, 81.175]
    estrus = [1103.821, 1534.872, 957.187, 225.609]
    x = np.arange(len(act))
    width = 0.3
    plt.subplot(2,2,1)
    plt.bar(x, normal, width, color='green', label='Nornal')
    plt.bar(x + width, estrus, width, color='blue', label='Estrus')
    plt.xticks(x + width / 2, act)
    plt.ylabel('score')
    plt.title('Dataset1')
    #plt.legend(loc='upper left')

    normal = [759.033,  1883.359, 929.713, 78.752]
    estrus = [1066.265, 1723.612, 791.807, 106.730]
    x = np.arange(len(act))
    width = 0.3
    plt.subplot(2,2,2)
    plt.bar(x, normal, width, color='green', label='Nornal')
    plt.bar(x + width, estrus, width, color='blue', label='Estrus')
    plt.xticks(x + width / 2, act)
    plt.ylabel('score')
    plt.title('Dataset2')
    plt.legend(bbox_to_anchor=(1,1), loc='upper left')


    normal = [872.661,  2041.804, 680.751, -44.073]
    estrus = [852.612, 2097.720, 647.235, 74.218]
    x = np.arange(len(act))
    width = 0.3
    plt.subplot(2,2,3)
    plt.bar(x, normal, width, color='green', label='Nornal')
    plt.bar(x + width, estrus, width, color='blue', label='Estrus')
    plt.xticks(x + width / 2, act)
    plt.ylabel('score')
    plt.title('Dataset3')


    normal = [633.169,  2058.924, 904.747, 7.748]
    estrus = [1230.079, 1681.378, 687.066, 98.663]
    x = np.arange(len(act))
    width = 0.3
    plt.subplot(2,2,4)
    plt.bar(x, normal, width, color='green', label='Nornal')
    plt.bar(x + width, estrus, width, color='blue', label='Estrus')
    plt.xticks(x + width / 2, act)
    plt.ylabel('score')
    plt.title('Dataset4')
    plt.show()

def bar_chart_normal():
    act = ['In_alleys', 'Rest', 'Eat', 'Activity level']

    data1 = [743.413, 1899.779, 950.424, 81.175]
    data2 = [759.033, 1883.359, 929.713, 78.752]
    data3 = [872.661, 2041.804, 680.751, -44.073]
    data4 = [633.169, 2058.924, 904.747, 7.748]
    
    x = np.arange(len(act))
    width = 0.2
    plt.bar(x, data1, width, color='green', label='Data1')
    plt.bar(x + width, data2, width, color='pink', label='Data2')
    plt.bar(x + width*2, data3, width, color='orange', label='Data3')
    plt.bar(x + width*3, data4, width, color='blue', label='Data4')
    plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    plt.xticks(x + 1.5*width , act)
    plt.ylabel('score')
    plt.title('Normal')
    
    plt.show()

def bar_chart_estrus():
    act = ['In_alleys', 'Rest', 'Eat', 'Activity level']

    data1 = [1103.821, 1534.872, 957.187, 225.609]
    data2 = [1066.265, 1723.612, 791.807, 106.730]
    data3 = [852.612 , 2097.720, 647.235, 74.218]
    data4 = [1230.079, 1681.378, 687.066, 98.663]
    
    x = np.arange(len(act))
    width = 0.2
    plt.bar(x, data1, width, color='green', label='Data1')
    plt.bar(x + width, data2, width, color='pink', label='Data2')
    plt.bar(x + width*2, data3, width, color='orange', label='Data3')
    plt.bar(x + width*3, data4, width, color='blue', label='Data4')
    plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    plt.xticks(x + 1.5*width , act)
    plt.ylabel('score')
    plt.title('Estrus')
    
    plt.show()

pie_chart()