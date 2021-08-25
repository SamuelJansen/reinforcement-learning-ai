import matplotlib.pyplot as plt


DATA = {
    0: {
        'winCount': 0,
        'loseCount': 30,
        'drawCount': 0
    },
    1: {
        'winCount': 0,
        'loseCount': 7,
        'drawCount': 23
    },
    2: {
        'winCount': 0,
        'loseCount': 7,
        'drawCount': 23
    },
    3: {
        'winCount': 0,
        'loseCount': 30,
        'drawCount': 0
    },
    4: {
        'winCount': 0,
        'loseCount': 30,
        'drawCount': 0
    }
}

GRID_WORLD = {
    0: {
        'winCount': 4,
        'loseCount': 2,
        'drawCount': 0
    },
    1: {
        'winCount': 4,
        'loseCount': 2,
        'drawCount': 0
    },
    2: {
        'winCount': 6,
        'loseCount': 0,
        'drawCount': 0
    },
    3: {
        'winCount': 6,
        'loseCount': 0,
        'drawCount': 0
    },
    4: {
        'winCount': 6,
        'loseCount': 0,
        'drawCount': 0
    },
    5: {
        'winCount': 6,
        'loseCount': 0,
        'drawCount': 0
    },
    6: {
        'winCount': 6,
        'loseCount': 0,
        'drawCount': 0
    },
    7: {
        'winCount': 6,
        'loseCount': 0,
        'drawCount': 0
    },
    8: {
        'winCount': 6,
        'loseCount': 0,
        'drawCount': 0
    },
    9: {
        'winCount': 6,
        'loseCount': 0,
        'drawCount': 0
    }
}

# X_WINS = [e['loseCount'] for e in DATA.values()]
# O_WINS = [e['winCount'] for e in DATA.values()]
# DRAWS = [e['drawCount'] for e in DATA.values()]
#
# # Plot lists and show them
# plt.plot(X_WINS, 'go-', label='X_WINS')
# plt.plot(O_WINS, 'b*--', label='O_WINS')
# plt.plot(DRAWS, 'r*-', label='DRAWS')
#
# # Plot axes labels and show the plot
# plt.xlabel('100 trainnings')
# plt.ylabel('event count')
# plt.legend()
# plt.show()


def printGraph(title, data):
    X_WINS = [e['loseCount'] for e in data.values()]
    O_WINS = [e['winCount'] for e in data.values()]
    DRAWS = [e['drawCount'] for e in data.values()]

    # Plot lists and show them
    plt.plot(X_WINS, 'go-', label='X wins')
    plt.plot(O_WINS, 'b*--', label='O wins')
    plt.plot(DRAWS, 'r*-', label='draws')

    # Plot axes labels and show the plot
    plt.title(title)
    plt.xlabel('trainning iteration - 100 trainnings')
    plt.ylabel('event count')
    plt.legend()
    plt.show()

# printGraph('test', DATA)
