import matplotlib.pyplot as plt


def printGraph(title, data, agent):
    winsCount = [e['winCount'] for e in data.values()]
    loseCount = [e['loseCount'] for e in data.values()]
    DRAWS = [e['drawCount'] for e in data.values()]

    # Plot lists and show them
    plt.plot(winsCount, 'b*--', label='wins')
    plt.plot(loseCount, 'go-', label='lose')
    plt.plot(DRAWS, 'r*-', label='draws')

    # Plot axes labels and show the plot
    plt.title(title + ' - ' + agent.getKey())
    plt.xlabel('trainning batch iteration | batch size: 100')
    plt.ylabel('event count')
    plt.legend()
    plt.show()
