{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.3"
    },
    "colab": {
      "name": "Project Clustering Blobs.ipynb",
      "provenance": []
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "K6uszEF-tg6Q"
      },
      "source": [
        "# Unsupervised Machine Learning - KMeans\n",
        "\n",
        "Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection. SVMs are one of the most robust prediction methods. \n",
        "\n",
        "Sources: \n",
        "[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html?highlight=kmeans#sklearn.cluster.KMeans), [wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fUB89AR6C5Br"
      },
      "source": [
        "![kmeans.png](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/52579/versions/9/screenshot.jpg)"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3s6IxHbEIGDC"
      },
      "source": [
        ""
      ],
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "V9gxYilRIGDD"
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn import datasets\n"
      ],
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "5lqqqeEcIGDD"
      },
      "source": [
        "# Take make moons in built dataset\n",
        "\n",
        "data_=datasets.make_moons(50)"
      ],
      "execution_count": 48,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YHE_oQTCIGDE",
        "outputId": "af7eb477-b72e-4f9f-b5c1-c6c63325828e",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "# check the dataset\n",
        "\n",
        "data_"
      ],
      "execution_count": 49,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(array([[-9.91444861e-01,  1.30526192e-01],\n",
              "        [ 6.12323400e-17,  1.00000000e+00],\n",
              "        [ 1.25881905e+00, -4.65925826e-01],\n",
              "        [ 1.86602540e+00, -3.33066907e-16],\n",
              "        [ 1.79335334e+00, -1.08761429e-01],\n",
              "        [ 6.17316568e-01, -4.23879533e-01],\n",
              "        [-5.00000000e-01,  8.66025404e-01],\n",
              "        [ 8.69473808e-01, -4.91444861e-01],\n",
              "        [ 3.91238571e-01, -2.93353340e-01],\n",
              "        [ 7.93353340e-01,  6.08761429e-01],\n",
              "        [ 2.92893219e-01, -2.07106781e-01],\n",
              "        [ 2.58819045e-01,  9.65925826e-01],\n",
              "        [ 5.00000000e-01,  8.66025404e-01],\n",
              "        [ 1.70710678e+00, -2.07106781e-01],\n",
              "        [ 1.38268343e+00, -4.23879533e-01],\n",
              "        [ 9.91444861e-01,  1.30526192e-01],\n",
              "        [-9.23879533e-01,  3.82683432e-01],\n",
              "        [-9.65925826e-01,  2.58819045e-01],\n",
              "        [ 5.00000000e-01, -3.66025404e-01],\n",
              "        [ 2.00000000e+00,  5.00000000e-01],\n",
              "        [-7.93353340e-01,  6.08761429e-01],\n",
              "        [ 1.33974596e-01,  0.00000000e+00],\n",
              "        [ 1.92387953e+00,  1.17316568e-01],\n",
              "        [ 1.00000000e+00,  0.00000000e+00],\n",
              "        [-6.08761429e-01,  7.93353340e-01],\n",
              "        [-1.30526192e-01,  9.91444861e-01],\n",
              "        [-1.00000000e+00,  1.22464680e-16],\n",
              "        [-3.82683432e-01,  9.23879533e-01],\n",
              "        [ 1.13052619e+00, -4.91444861e-01],\n",
              "        [-7.07106781e-01,  7.07106781e-01],\n",
              "        [ 7.07106781e-01,  7.07106781e-01],\n",
              "        [ 0.00000000e+00,  5.00000000e-01],\n",
              "        [ 1.00000000e+00, -5.00000000e-01],\n",
              "        [ 3.82683432e-01,  9.23879533e-01],\n",
              "        [ 6.08761429e-01,  7.93353340e-01],\n",
              "        [ 1.99144486e+00,  3.69473808e-01],\n",
              "        [ 1.96592583e+00,  2.41180955e-01],\n",
              "        [ 3.40741737e-02,  2.41180955e-01],\n",
              "        [-2.58819045e-01,  9.65925826e-01],\n",
              "        [ 1.30526192e-01,  9.91444861e-01],\n",
              "        [ 1.50000000e+00, -3.66025404e-01],\n",
              "        [ 8.66025404e-01,  5.00000000e-01],\n",
              "        [ 9.23879533e-01,  3.82683432e-01],\n",
              "        [-8.66025404e-01,  5.00000000e-01],\n",
              "        [ 8.55513863e-03,  3.69473808e-01],\n",
              "        [ 7.61204675e-02,  1.17316568e-01],\n",
              "        [ 2.06646660e-01, -1.08761429e-01],\n",
              "        [ 1.60876143e+00, -2.93353340e-01],\n",
              "        [ 7.41180955e-01, -4.65925826e-01],\n",
              "        [ 9.65925826e-01,  2.58819045e-01]]),\n",
              " array([0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1,\n",
              "        1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,\n",
              "        1, 1, 1, 1, 1, 0]))"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 49
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eRSFx6IrIGDF"
      },
      "source": [
        "# create input dataframe\n",
        "\n",
        "inputData = pd.DataFrame(data=data_[0])"
      ],
      "execution_count": 55,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VvAL1WJFIGDG",
        "outputId": "6c3abd33-4c02-427f-e2b3-6a36e3f30016",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 203
        }
      },
      "source": [
        "inputData.head()"
      ],
      "execution_count": 51,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "      <th>1</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>-9.914449e-01</td>\n",
              "      <td>1.305262e-01</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>6.123234e-17</td>\n",
              "      <td>1.000000e+00</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1.258819e+00</td>\n",
              "      <td>-4.659258e-01</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1.866025e+00</td>\n",
              "      <td>-3.330669e-16</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>1.793353e+00</td>\n",
              "      <td>-1.087614e-01</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "              0             1\n",
              "0 -9.914449e-01  1.305262e-01\n",
              "1  6.123234e-17  1.000000e+00\n",
              "2  1.258819e+00 -4.659258e-01\n",
              "3  1.866025e+00 -3.330669e-16\n",
              "4  1.793353e+00 -1.087614e-01"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 51
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JTWRMvLZIGDI",
        "outputId": "1a7ce493-b8b1-4115-b738-728e4b35edfd",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 203
        }
      },
      "source": [
        "# create output dataframe\n",
        "\n",
        "outputData = pd.DataFrame(data=data_[1])\n",
        "outputData.head()"
      ],
      "execution_count": 54,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   0\n",
              "0  0\n",
              "1  0\n",
              "2  1\n",
              "3  1\n",
              "4  1"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 54
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "gtvIo2RKIGDJ",
        "outputId": "87ff4ee0-4c40-412c-8cf3-773aa205f375",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        }
      },
      "source": [
        "# create a scatter plot for inputData set\n",
        "\n",
        "plt.scatter(inputData[0],inputData[1],)"
      ],
      "execution_count": 57,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.collections.PathCollection at 0x7ff5b28a4f10>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 57
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xzS4s5hTIGDK",
        "outputId": "6972b7aa-3411-4c4b-8d52-e86163c05290",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        }
      },
      "source": [
        "# create a scatter plot for inputData set with outputData color\n",
        "\n",
        "plt.scatter(inputData[0],inputData[1],c=outputData)"
      ],
      "execution_count": 65,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.collections.PathCollection at 0x7ff5b2706f10>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 65
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "M3_TfCHXIGDL"
      },
      "source": [
        "\n",
        "from sklearn import metrics"
      ],
      "execution_count": 80,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IlgenF6VIGDL",
        "outputId": "4b7b633a-4b0e-479b-d10b-1a89c0520e00",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "# Call the sklearn Kmeans and make a model with 200 samples\n",
        "\n",
        "#model_fit\n",
        "from sklearn.cluster import KMeans\n",
        "\n",
        "model=KMeans(n_clusters=5)\n",
        "model.fit(inputData)\n"
      ],
      "execution_count": 81,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,\n",
              "       n_clusters=5, n_init=10, n_jobs=None, precompute_distances='auto',\n",
              "       random_state=None, tol=0.0001, verbose=0)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 81
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dQxPm2WiIGDM",
        "outputId": "566bbd20-5679-4585-caff-72048e75ae74",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "# check for labels\n",
        "model.labels_\n"
      ],
      "execution_count": 75,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([1, 4, 3, 2, 2, 3, 4, 3, 3, 0, 3, 4, 0, 2, 2, 0, 1, 1, 3, 2, 1, 3,\n",
              "       2, 3, 1, 4, 1, 4, 3, 1, 0, 4, 3, 4, 0, 2, 2, 4, 4, 4, 2, 0, 0, 1,\n",
              "       4, 4, 3, 2, 3, 0], dtype=int32)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 75
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "H-zE7M7lIGDM"
      },
      "source": [
        ""
      ],
      "execution_count": None,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "22fBru-tIGDN",
        "outputId": "88b22899-4a1b-4fc0-8b49-94d568fdb988",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        }
      },
      "source": [
        "# call metrics and check silhoutte score\n",
        "plt.scatter(inputData[0],inputData[1],c =model.labels_)\n"
      ],
      "execution_count": 82,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.collections.PathCollection at 0x7ff5b2556710>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 82
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "RR6CaZ1PIGDN"
      },
      "source": [
        "# create a scatter plot for inputData set with model labels color\n",
        "\n"
      ],
      "execution_count": 77,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_wbZtm1IIGDO"
      },
      "source": [
        "#### finding right number of cluster"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ncq-IdIgIGDO"
      },
      "source": [
        "cluster_range = range(1, 20)\n",
        "error_list = []\n",
        "\n",
        "for i in cluster_range:\n",
        "    model = KMeans(n_clusters=i)\n",
        "    model.fit(inputData)\n",
        "    res = model.inertia_\n",
        "    error_list.append(res)"
      ],
      "execution_count": 78,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eYfIL5LfIGDO",
        "outputId": "599d6429-e795-4a58-e70e-8d88b13d24d6",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        }
      },
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "plt.plot(cluster_range, error_list, marker = \"o\", color = \"g\", markersize = 10)\n",
        "plt.xlabel(\"Cluster Range\")\n",
        "plt.ylabel(\"IntraCluster Sum\")\n",
        "plt.title(\"KMeans\")\n",
        "plt.show()"
      ],
      "execution_count": 79,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAX4AAAEWCAYAAABhffzLAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3df5wVZd3/8dfn7J4Flt0FVlB+KQi76q3eKyqSqJWpmZiZdWOZgZQaGnmXRo+y7O5rdpd1J5mlRYqWaZaJGoqSeatld4oKqPgjc1mEBAUXlx8LC+yy+/n+cWbpsOyeHWDPmbNn3s/H4zzOmZlrZj7nsHxm5pprrsvcHRERiY9E1AGIiEhuKfGLiMSMEr+ISMwo8YuIxIwSv4hIzCjxi4jEjBK/iEjMKPFLLJjZCjM7LW36PDNbb2bvNzM3s+c7lB9sZs1mtiLnwYpkmRK/xI6ZTQNuAj4MrAxml5rZkWnFzgfeyHVsIrmgxC+xYmaXALOAD7n7U2mL7gCmpU1fAPy6w7rDzexeM6s3szfM7ItpyyaY2dNmtsHM3jazG82sJG25m9mlZlYblLnJzCxYVmVmfzGzjWa2zszuzsZ3F2mnxC9x8nngGuBUd1/UYdmdwHlmVmRmhwNlwDPtC80sATwIvAiMAE4FLjezDwVFWoErgMHAxGD5jA77OAs4DqgBPgG0r/sd4E/AIGAk8NN9/qYiGSjxS5x8EFgIvNTJslXAP4DTSJ3t39Fh+XHAEHe/xt2b3X05cAtwHoC7L3b3he6+w91XAL8A3t9hG9939w3u/k/gCWBcML8FGAUMd/dt7v5/+/pFRTJR4pc4+TxwCDCnvZqlg18DnwE+xe6JfxQwPKim2WBmG4BvAAcAmNkhZjbfzNaY2Sbge6TO/tOtSfvcROqqAuCrgAHPmtkrZnbhXn9DkRCU+CVO1pKqgnkv8LNOlt9L6obv8uCsPN2bwBvuPjDtVe7uZwbLfw68BlS7ewWpg0JnB5fduPsad/+cuw8HLgF+ZmZVe/ztREJS4pdYcfe3SCX/M8zs+g7LtgCnABd3suqzQKOZfc3M+gX3Ao40s+OC5eXAJmCzmR1G6uoiFDM718xGBpPrAQfa9uiLiewBJX6JneBs/hRgMnBth2WL3L2uk3VaSd2cHUeqmec6YA4wICjyFVJNQBtJ1f3vScuc44BnzGwz8ADwpeAegkhWmAZiERGJF53xi4jEjBK/iEjMKPGLiMSMEr+ISMwURx1AGIMHD/bRo0dHHYaISK+yePHide4+pOP8XpH4R48ezaJFHbtWERGRTMxsZWfzVdUjIhIzSvwiIjGjxC8iEjMFmfjrGuqY8dAMKq6tIPHtBBXXVjDjoRnUNez2JL6ISOwUXOJfULuAmtk1zFkyh8bmRhynsbmROUvmUDO7hgW1C6IOUUQkUgWV+Osa6ph8z2SaWppoaWvZZVlLWwtNLU1MvmeyzvxFJNYKKvHPenoWLa0tGcu0tLZw/cLrM5YRESlkWU38ZrbCzF4ysxfMbFEwr9LMHg0GnX7UzAb11P7uXHrnbmf6HbW0tXDH0o6DK4mIxEcuzvg/4O7j3H18MH0l8Ji7VwOPBdM9YnPz5h4tJyJSiKKo6vkocHvw+XbgnJ7acFlJWfeF9qCciEghynbid+BPZrbYzKYH8w5w97eDz2sIBqvuyMymm9kiM1tUX18famdTaqaQTCQzlkkmkkytmRouehGRApTtxH+Sux8DTAK+YGbvS1/oqeG/Oh0CzN1vdvfx7j5+yJDd+hjq1MyJM0kWdZP4i5JccfwVobYnIlKIspr43X118P4OcD8wAVhrZsMAgvd3emp/YyvHMvfcuZQmS3c7808mkpQmS5l77lzGVo7tqV2KiPQ6WUv8ZtbfzMrbPwOnAy+TGkx6WlBsGjCvJ/c7qXoSSy9dyvRjp1PRpwKAkqISph87naWXLmVS9aSe3J2ISK+TtcHWzWwMqbN8SHX/fJe7f9fM9gN+DxwErAQ+4e4NmbY1fvx439tumY/42REcut+h3PfJ+/ZqfRGR3srMFqe1qNwpa/3xu/ty4KhO5r8LnJqt/XZUVVlFbUNtrnYnIpL3CurJ3c5UDaqirqGONm+LOhQRkbxQ+Im/soqtO7byduPb3RcWEYmBWCR+gGUNyyKOREQkPyjxi4jETMEn/oMGHEQykVTiFxEJFHziL0oUMWbQGLXsEREJFHzih1R1j874RURSYpX4s/WwmohIbxKbxL+lZQtrt6yNOhQRkcjFJvGDWvaIiEBMEn91ZTWgxC8iAjFJ/KMGjqI4UazELyJCTBJ/caKY0QNHq0mniAgxSfygJp0iIu3ik/gHqUmniAjEKfFXVrFp+ybWNa2LOhQRkUjFKvGDWvaIiMQm8VfvpyadIiIQo8Q/euBoEpZQyx4Rib3YJP6SohJGDRilM34Rib3YJH5Qk04REVDiFxGJndgl/vXb1tOwtSHqUEREIhOrxK/O2kREYpb41ZZfRCRmif/gQQdjGLXvqkmniMRXrBJ/3+K+HDjgQJat1xm/iMRXrBI/qGWPiEj8Ev8gJX4RibfYJf7q/apZ17SODds2RB2KiEgksp74zazIzJ43s/nB9MFm9oyZLTOzu82sJNsxpGtv2VPXUJfL3YqI5I1cnPF/Cfh72vQPgOvdvQpYD1yUgxh2ak/86qxNROIqq4nfzEYCHwbmBNMGnALMDYrcDpyTzRg6GjNoDKC2/CISX9k+4/8x8FWgLZjeD9jg7juC6VXAiM5WNLPpZrbIzBbV19f3WEClyVJGlI9Q4heR2Mpa4jezs4B33H3x3qzv7je7+3h3Hz9kyJAejU1NOkUkzrJ5xn8icLaZrQB+R6qK5wZgoJkVB2VGAquzGEOnlPhFJM6ylvjd/evuPtLdRwPnAY+7+6eBJ4DJQbFpwLxsxdCV6spq1m5ZS+P2xlzvWkQkclG04/8a8GUzW0aqzv/WXAegztpEJM6Kuy+y79z9z8Cfg8/LgQm52G9X0hP/0cOOjjIUEZGci92TuwBjK8cCOuMXkXiKZeIvKyljaNlQJX4RiaVYJn4IWvaoe2YRiaHYJv7qymqd8YtILMU28VdVVvFW41tsad4SdSgiIjkV68QPULdevXSKSLzEPvGrukdE4ia2iX/sIDXpFJF4im3iH9B3AENKhyjxi0jsxDbxgzprE5F4inXir95PTTpFJH5infirBlXx5qY32dqyNepQRERypttO2sysiNTwiaPTy7v7j7IXVm60t+xZvn45R+x/RMTRiIjkRpgz/geBz5DqQrk87dXrqUmniMRRmG6ZR7p7TdYjiYASv4jEUZgz/gVmdnrWI4nAoH6DqOxXqcQvIrES5ox/IXC/mSWAFsAAd/eKrEaWI9WV1eqlU0RiJcwZ/4+AiUCpu1e4e3mhJH1QW34RiZ8wif9N4GV392wHE4Wqyir+ufGfbN+xPepQRERyIkxVz3Lgz2a2ANiZHQuhOSekEn+bt/HGhjc4bPBhUYcjIpJ1Yc743wAeA0oosOacoJY9IhI/3Z7xu/u3cxFIVJT4RSRuwjy5+wSwW/2+u5+SlYhybL9++zGgzwAlfhGJjTB1/F9J+9wX+A9gR3bCyT0zU2dtIhIrYap6FneY9TczezZL8USiqrKKZ1cX1FcSEelStzd3zawy7TXYzD4EDMhBbDlTNaiKFRtW0NzaHHUoIiJZF6aqZzGpOn4jVcXzBnBRNoPKtfYmnSs3rKR6v+qowxERyaowVT0H5yKQKKW37FHiF5FC12VVj5kdZ2ZD06YvMLN5ZvYTM6vMTXi5oSadIhInmer4fwE0A5jZ+4DvA78GNgI3Zz+03Nm///6Ul5Qr8YtILGSq6ily94bg8yeBm939XuBeM3sh+6HljpmlOmtTL50iEgOZzviLzKz9wHAq8HjasjAPfvU1s2fN7EUze8XMvh3MP9jMnjGzZWZ2t5mV7H34Paeqsorad2ujDkNEJOsyJf7fAn8xs3nAVuCvAGZWRaq6pzvbgVPc/ShgHHCGmR0P/AC43t2rgPXkSQuhqsoq3tjwBjvaCubZNBGRTnWZ+N39u8BM4FfASWndMieA/+xuw56yOZhMBi8HTgHmBvNvB87Zq8h7WFVlFTvadvDPjf+MOhQRkazK+ACXuy909/vdfUvavNfdfUmYjZtZUXA/4B3gUaAO2ODu7afVq4ARXaw73cwWmdmi+vr6MLvbJ2rZIyJxEaZb5r3m7q3uPg4YCUwAQnd47+43u/t4dx8/ZMiQrMXYTolfROIiq4m/nbtvAJ4gNYTjwLSbxiOB1bmIoTvDyoZRmixV4heRgpcx8QdVNU/szYbNbIiZDQw+9wM+CPyd1AFgclBsGjBvb7bf09qbdNY2qGWPiBS27ur4W4E2M9ubTtmGAU+Y2VLgOeBRd58PfA34spktA/YDbt2LbWeFBl4XkTgI00nbZuAlM3sUSL/J+8VMK7n7UuDoTuYvJ1Xfn3eqBlUx//X5tLa1UpQoijocEZGsCJP47wteBa+qsorm1mZWbVrFqIGjog5HRCQrwvTOeXtQR3+Qu/8jBzFFJr1ljxK/iBSqMAOxfAR4AfhjMD3OzB7IdmBRaO+SWfX8IlLIwjTnvJpUnfwGAHd/ARiTxZgiM7x8OH2L+yrxi0hBC5P4W9y9Y988bdkIJmoJSzB20Fg16RSRghbm5u4rZnY+qd46q4EvAk9lN6zoqEmniBS6MGf8/wkcQaq3zbtI9cz5pWwGFaWqyirq1tfR5gV5USMiEirxf9jdr3L344LXN4Gzsx1YVKoqq9i2YxtvNb4VdSgiIlkRJvF/PeS8glBdqZY9IlLYuqzjN7NJwJnACDP7SdqiCqBgRytJb8t/8uiTow1GRCQLMt3cfQtYRKpaZ3Ha/EbgimwGFaWRFSMpKSrRMIwiUrC6TPzu/iLwopnd5e4tAGY2CDjQ3dfnKsBcK0oUMWbQGA28LiIFK0wd/6NmVmFmlcAS4BYzuz7LcUVKTTpFpJCFSfwD3H0T8HHg1+7+HuDU7IYVrapBqcT/r2GGRUQKR5jEX2xmw4BPAPOzHE9eqKqsoqmliTWb10QdiohIjwuT+K8BHgGWuftzZjYGKOg7n+qsTUQKWbeJ393vcfcad58RTC939//IfmjR0cDrIlLIuu2rx8x+CexW2e3uF2Ylojxw0ICDKE4Uq7M2ESlIYTppS6/X7wt8jFQb/4JVnCjm4IEH64xfRApSmBG47k2fNrPfAv+XtYjyhJp0ikihCnNzt6NqYP+eDiTftCd+NekUkUITpo6/kVQdvwXva4CvZTmuyFVXVtPY3Eh9Uz379y/445yIxEiYqp7yXASSb9Jb9ijxi0ghydQ75zGZVnT3JT0fTv5oT/y179ZywoEnRByNiEjPyXTGPyvDMgdO6eFY8sqogaMosiLd4BWRgpOpd84P5DKQfFNSVMKogaPUS6eIFJwuW/WY2RQzm9rJ/KnB4OsFT006RaQQZWrO+Z/A/Z3Mvw+YmZ1w8kvVoCpq361Vk04RKSiZEn/S3Td3nOnuW4Bk9kLKD3UNdby49kU2bt9I0TVFVFxbwYyHZlDXUBd1aCIi+yRT4u9nZv07zjSzcqAkeyFFb0HtAmpm17Bw1UIAHKexuZE5S+ZQM7uGBbULIo5QRGTvZUr8twJzzWxU+wwzGw38LliWkZkdaGZPmNmrZvaKmX0pmF9pZo+aWW3wPmjfvkLPqmuoY/I9k2lqaaLVW3dZ1tLWQlNLE5PvmawzfxHptbpM/O5+HTAPeNLM3jWzBuAvwHx3/2GIbe8AZrr74cDxwBfM7HDgSuAxd68GHgum88asp2fR0tqSsUxLawvXLyzo0SdFpIBl7KvH3We7+yhgNDDK3Ue5+8/DbNjd325/yMvdG4G/AyOAjwK3B8VuB87Zy9iz4s6ld9LS1k3ib2vhjqV35CgiEZGeFaZbZoD3AUeYWd/2Ge5+TdidBFVERwPPAAe4+9vBojXAAWG3kwubm3e7n71P5URE8k23vXOa2Wzgk6SadxpwLjAq40q7rl8G3AtcHgzavpOn2kl22lbSzKab2SIzW1RfXx92d/usrKSsR8uJiOSbMN0yn+DuFwDr3f3bwETgkDAbN7MkqaT/G3e/L5i9Nhi8neD9nc7Wdfeb3X28u48fMmRImN31iCk1U0gmMrdWTSaSTK3Z7dk2EZFeIUzi3xa8N5nZcKAFGNbdSmZmpFr//N3df5S26AFgWvB5GqkbyHlj5sSZJIu6SfxFSa44/oocRSQi0rPCJP4HzWwg8ENgCbACuCvEeicCU4FTzOyF4HUm8H3gg2ZWC5wWTOeNsZVjmXvuXEqTpZ2e+SdIcPfkuxlbOTaC6ERE9l3GxG9mCVJNLzcEQzCOAg5z9291t2F3/z93N3evcfdxwethd3/X3U9192p3P83dG3rou/SYSdWTWHrpUqYfO52KPhUkLEFFnwo+NPZDtNHGK++8EnWIIiJ7zbrrh8bMnnf3o3MUT6fGjx/vixYtijIEANydyfdM5qHXH+LFS1/k0MGHRh2SiEiXzGyxu4/vOD9MVc9jZvYfQZ19rJkZN515E6XJUi584EJa21q7X0lEJM+ESfyXAPcA281sk5k1mtmm7lYqVEPLhvLjM37MU28+xU3P3RR1OCIie6zbxO/u5e6ecPcSd68IpityEVy+mlozlUlVk/j6Y19n+frlUYcjIrJHwjzA9ViYeXFiZvzirF9QZEV87sHPqb9+EelVMo3A1dfMKoHBZjYo6FWzMuh+YUSuAsxXBw44kOtOv47H33icOUvmRB2OiEhomc74LwEWA4cF7+2vecCN2Q8t/33umM/xgdEfYOafZrJq06qowxERCSVTt8w3uPvBwFfcfYy7Hxy8jnJ3JX5SVT63fOQWWr2VS+ZfoiofEekVwtzc/amZnWBm55vZBe2vXATXG4ytHMv3TvkeD9c+zG9e+k3U4YiIdCvMzd07gOuAk4DjgtduDwTE2WUTLmPiyIl8ccEXWbN5TdThiIhkFKY//vHA4a56jC4VJYq47aO3MW72OC57+DLmfmJu1CGJiHQpzANcLwNDsx1Ib3fY4MO4+uSruffv9zL3VSV+EclfYRL/YOBVM3vEzB5of2U7sN7oKyd8hWOGHcMXHv4C7za9G3U4IiKdClPVc3W2gygUxYlibjv7NsbfMp7LH7mcOz6mcXlFJP90m/jd/S+5CKRQHDX0KL5x0je45slrOO+I8/jwIR+OOiQRkV102S2zmTXS+Xi4Rmq43Jz115Mv3TKH1dzazLE3H0v9lnrOOuQsfv/K79ncvJmykjKm1Exh5sSZGshFRLJuj7tlbu+MrZNX7Dtp605JUQkXH30xa7es5bbnb6OxuRHHaWxuZM6SOdTMrmFB7YKowxSRmApzc1f2UF1DHd94/BsAeIeLppa2Fppamph8z2TqGuqiCE9EYk6JPwtmPT2LltaWjGVaWlu4fuH1OYpIRORflPiz4M6ld9LS1k3ib2vhjqVq9SMiuafEnwWbmzf3aDkRkZ6kxJ8FZSVlPVpORKQnKfFnwZSaKSQTyYxlkokkU2um5igiEZF/UeLPgpkTZ5Isypz4ixJFXHH8FTmKSETkX5T4s2Bs5VjmnjuX0mTpbmf+xYnUw9JlyTL6JftFEZ6IxJwSf5ZMqp7E0kuXMv3Y6VT0qSBhCSr6VHDJsZcw77x5bGvdxofv+jCN2xujDlVEYqbLLhvySW/rsiGMPy77I2fddRanjTmNBz/1YLdVQyIie2qPu2yQ7Dqj6gxmnzWbR+oeYcZDMzRer4jkTJhumSVLLj7mYlZuWMl///W/GT1wNFe976qoQxKRGFDij9g1H7iGlRtX8s0nvsmogaOYUjMl6pBEpMAp8UfMzJhz9hxWbVrFhfMuZHj5cE45+JSowxKRApa1On4zu83M3jGzl9PmVZrZo2ZWG7wPytb+e5OSohLu++R9HLLfIXz87o/z8jsvd7+SiMheyubN3V8BZ3SYdyXwmLtXA48F0wIM7DuQhz/9MP1L+nPmb87krca3og5JRApU1hK/uz8JNHSY/VHg9uDz7cA52dp/b3TQgIN46PyHWL9tvdr4i0jW5Lo55wHu/nbweQ1wQFcFzWy6mS0ys0X19fW5iS4PjBs6jrnnzuWltS9x7j3n8lr9a8x4aAYV11aQ+HaCimsrmPHQDA3iIiJ7LasPcJnZaGC+ux8ZTG9w94Fpy9e7e7f1/IX4AFd3bl1yKxc/eDHFiWIM26V//2QiSbIoydxz5zKpelKEUYpIPsuXB7jWmtmwIKBhwDs53n+vcfLok0kmkuxo27HboC4avlFE9kWuE/8DwLTg8zRgXo7332vMenpWt2U0fKOI7I1sNuf8LfA0cKiZrTKzi4DvAx80s1rgtGBaOqHhG0UkW7L2AJe7f6qLRadma5+FRMM3iki2qJO2PKXhG0UkW5T481SY4RsBWttauenZm9i2Y1sOohKRQqDEn6fCDN/Yp6gPhw4+lMsWXMaYG8Zww8IbaGpp2q1cXUOdngUQkZ2U+PNUpuEbk4kkpclS7v/k/Sz63CIev+BxDh18KJc/cjljbhjDrKdmsaV5CwALahdQM7uGOUvm0NjciOM0NjcyZ8kcambXsKB2QRRfT0QipBG48lxdQx3XL7yeO5bewebmzZSVlDG1ZipXHH8FYyvH7lL2yZVP8p0nv8P/Lv9fBpcO5rNHfZabFt3U6VVAu9JkKUsvXbrbtkSk9+vqAS4l/gL01JtP8Z0nv8Mfl/2x27LJRJLpx07nxjNvzEFkIpJL+fLkruTACQeewIJPL6B/sn+3ZfUsgEj8KPEXsExVPOn0LIBIvCjxFzA9CyAinVHiL2BhnwUYNWAUqzetzkFEIpIPlPgLWJhnAYqtmFfrX2XMT8bw+fmfZ+WGlTmKTkSiosRfwMI8C/DApx5g2ReX8dlxn+XW52+l6qdVXDTvIpY1LNulvB4CEykcas4ZA2GfBVi1aRX/87f/4ZYlt9Dc2sz5/34+V733Kt5Y/waT75lMS2uLBoQR6UXUjl9CW7N5Ddc9dR0/X/RzmlqaKLIiWr21y/J6CEwkP6kdv4Q2tGwo151+HSu+tIKjhx6dMelDuAFhVFUkkj+U+KVLQ/oP2a2uvzPdPQSm/oJE8osSv2QU9uGuTds3cd7c85j11CyeXPnkzvXqGuqYfM9kmlqaNHawSJ7I2ghcUhjKSspobG7stlxxopinVz3N3a/cDUDCEvzb4H9jR9uObscKaK8qUn9BIrmhM37JKMxDYMlEkkuOvYSVl69kzcw1zP/UfP7rff/FqIGjeP3d12nztozrh+0vSPcJRHqGWvVIRnUNddTMrtnrrp0T307ghPsbu/+T9zNhxASGlw/fbdmC2gVqUiqyh9ScU/baviTdimsrQlUVpRtZMZIJIyYwYfgE3jPyPQzqO4gTbjuhR8YVqGuoY9bTs7hz6Z07n2mYUjOFmRNnqjmqFBwlftknezIgTLoZD81gzpI5u93YTZdMJLnw6AuZdtQ0nl39LM+sfoZnVz9L3frwVThhxhXQVYPEjRK/RGJfqorebXqXZ1c/yzl3n0Nza3O3++pX3I/5589n1IBRjKwYSZ/iPj0SR2ffSVcN0hso8Utk9vVMe0/uE7QzjKFlQxk1cBQHDTiI19e9ztJ3lma80ZzLqwYdPCQXlPglUntbVQTh7xOUlZQx77x5/HPjP1m5YSUrN65Mfd64MtSDaAB9i/tyx8fuYET5CIaXD2dY+TBKikp2foeeuGrQwUNyRYlfeq2w9wkyna3vzVVDu8GlgxlRPoKGrQ2s2rQq43a6i6MQDx75sg3ZnRK/9Fo9kSzDXjWUl5Tz18/+lbca39rltbpxNQ/VPtTtMwmQenjt5NEnM7h0MIP7DU69B6/fvvRbFixbwA7f0eX6vengkS/bAB2AOqPEL73aviaHXF81nHjgiaxrWse6pnU0bG3Y46uNZCLJpeMvpbyknIo+FVT0qaC8T+rzrUtu5eHahyM/eOTLNiB/DkD5dvBR4pdeb1/uE+TyqqGiTwUbr9y4c7q1rZX129azrmkdh990eOiDwMC+A9m0fVOoq4zOFFkRHzn0I5SXlKdeff71/ofX/sCTK5/M2PNqdwePnjiY9sQ28uUAlC8Hn3RK/BJ7+XDVsKcHD3dn646tbNq+icbtjWzavonjbjku9MHjyP2PpHF7I43NjTRub8wYe2cM4+BBB9M/2Z/+Jf13eZ/76txQ2+tX3I+bP3IzCUvs8iqyIj5936fZumNrt9soLylnzVfW0Le4LwnbtaeZfDgA5cvBp6OuEr86aZPYmFQ9iaWXLt3rq4aZE2dy+4u3Z04ORUmuOP6KLpdPqZkSKsFMrZkKgJlRmiylNFnK0LKhQPiO8yr6VPDS51/aZd72HdtpbG5k/x/uH+rg4TgTR05kS8sWtjRvYUvLFuqb6tnSvCX0QWTrjq1MvX9qqLJdaWxupP/3+gPQp6gPpclS+iX70a+4H29seCNUf1A3L76Z7Tu2pw46iaKdB5+EJbj1+Vu7/T4tbS388vlfcvahZ9O3uC/9ivvRL9mPvsV9+c5fvkNLazfrd9MZ4aynZ+3zNsKK5IzfzM4AbgCKgDnu/v1M5XXGL/liX68aeuKsLoorj33ZRllJGc9f8jxt3kabt9Ha1rrz84m3nciWli3dbqNvcV+ufv/VbN2xlaaWJra2bGXrjtTrrpfu6nb9dsPLh+/cf6u37oxnT7sV2VuGcUDZASQsgWE7r37MjJUbVoY6GGf6N9ltf/lyxm9mRcBNwAeBVcBzZvaAu7+a61hE9tS+XjWMrRzL3HPndnvwyLSdKK489mUb046aRlVlVafLLzjqglDbuOjoi/jaSV/rdPmD/3gw9EFs9ZdXd74s5EGsf7I/j0x5hG07tqUOPC1b2bZjGxf84YJu14XUFdRHDvkI7p46+NG28/OKDStCbSPsGBmZ5PyM38wmAle7+4eC6a8DuPu1Xa2jM34pNPtyoxry48ojX7aRD3X8ubyC6okz/ij64x8BvJk2vSqYtwszm25mi8xsUX19fc6CE8mFsZVjufHMG9l45UZav9XKxis3cuOZN4a+add+5TH92OlU9NxXDKQAAAiwSURBVKkgYQkq+lQw/djpLL10abctP9qvPEqTpbuNt5BMJClNlnZ75ZEv25g5cSbJom7GjOjmCmhftxF23IrurqD2dRthRXHGPxk4w90vDqanAu9x98u6Wkdn/CLZsa9XHvmyjaibUubL1U9HedOcU1U9IpINUR+Aoj74dCafEn8x8DpwKrAaeA44391f6WodJX4R6Q2iPvh0lDeJPwjmTODHpJpz3ubu381UXolfRGTP5U1zTgB3fxh4OIp9i4jEXRStekREJEJK/CIiMdMrOmkzs3pgZdRxZDAYWBd1ECH1llgVZ8/qLXFC74m1N8Q5yt2HdJzZKxJ/vjOzRZ3dQMlHvSVWxdmzekuc0Hti7S1xdkZVPSIiMaPELyISM0r8PePmqAPYA70lVsXZs3pLnNB7Yu0tce5GdfwiIjGjM34RkZhR4hcRiRkl/pDM7EAze8LMXjWzV8zsS52UOdnMNprZC8HrW1HEGsSywsxeCuLYraMjS/mJmS0zs6VmdkwEMR6a9lu9YGabzOzyDmUi+U3N7DYze8fMXk6bV2lmj5pZbfA+qIt1pwVlas1sWgRx/tDMXgv+Xe83s4FdrJvxbyRHsV5tZqvT/n3P7GLdM8zsH8Hf65URxHl3WowrzOyFLtbN6W+619xdrxAvYBhwTPC5nFQPo4d3KHMyMD/qWINYVgCDMyw/E1gAGHA88EzE8RYBa0g9cBL5bwq8DzgGeDlt3v8AVwafrwR+0Ml6lcDy4H1Q8HlQjuM8HSgOPv+gszjD/I3kKNarga+E+NuoA8YAJcCLHf/vZTvODstnAd/Kh990b1864w/J3d929yXB50bg73Qyclgv8lHg156yEBhoZsMijOdUoM7d8+IJbXd/EmjoMPujwO3B59uBczpZ9UPAo+7e4O7rgUeBM3IZp7v/yd13BJMLgZHZ2v+e6OI3DWMCsMzdl7t7M/A7Uv8WWZEpTjMz4BPAb7O1/1xQ4t8LZjYaOBp4ppPFE83sRTNbYGZH5DSwXTnwJzNbbGbTO1keagjMHDqPrv8z5ctveoC7vx18XgMc0EmZfPtdLyR1ZdeZ7v5GcuWyoFrqti6qz/LpN30vsNbda7tYni+/aUZK/HvIzMqAe4HL3X1Th8VLSFVVHAX8FPhDruNLc5K7HwNMAr5gZu+LMJaMzKwEOBu4p5PF+fSb7uSp6/q8bgttZlcBO4DfdFEkH/5Gfg6MBcYBb5OqRslnnyLz2X4+/KbdUuLfA2aWJJX0f+Pu93Vc7u6b3H1z8PlhIGlmg3McZnssq4P3d4D7SV0up1sNHJg2PTKYF4VJwBJ3X9txQT79psDa9uqw4P2dTsrkxe9qZp8BzgI+HRykdhPibyTr3H2tu7e6extwSxcx5MtvWgx8HLi7qzL58JuGocQfUlC3dyvwd3f/URdlhgblMLMJpH7fd3MX5c44+ptZeftnUjf7Xu5Q7AHggqB1z/HAxrRqjFzr8iwqX37TwANAeyudacC8Tso8ApxuZoOCaovTg3k5Y2ZnAF8Fznb3TkfuDvk3knUd7it9rIsYngOqzezg4OrwPFL/Frl2GvCau6/qbGG+/KahRH13ube8gJNIXdovBV4IXmcClwKXBmUuA14h1epgIXBCRLGOCWJ4MYjnqmB+eqwG3ESqtcRLwPiIYu1PKpEPSJsX+W9K6kD0NtBCqk75ImA/4DGgFvhfoDIoOx6Yk7buhcCy4PXZCOJcRqpOvP3vdHZQdjjwcKa/kQhivSP4+1tKKpkP6xhrMH0mqZZ0ddmOtbM4g/m/av+7TCsb6W+6ty912SAiEjOq6hERiRklfhGRmFHiFxGJGSV+EZGYUeIXEYkZJX7plYL2/b8zs7rg8fiHzewQMxud3qviHm7zM2Y2fB/j+oyZ1Qe9M75mZlfsy/ZEskGJX3qd4IGu+4E/u/tYdz8W+Dqd952zJz5Dql32nsRS3Mnsu919HHAicJWZHdhJGZHIKPFLb/QBoMXdZ7fPcPcX3f2v6YWCs+8b06bnW6p//yIz+5WZvRz0nX6FmU0m9SDWb4Kz9X5mdqyZ/SW4ongkrbuGP5vZj4P+1ncblyEtpndJPUzVvt63zOy5YL83pz2R/Gcz+4GZPWtmr5vZe4P5pWb2e0uNAXG/mT1jZuODZaeb2dNmtsTM7gn6kBIJRYlfeqMjgcX7sP44YIS7H+nu/w780t3nAotI9W0zjlTnZj8FJgdXFLcB303bRom7j3f3LjsVM7ODgL6knkoFuNHdj3P3I4F+pPrSaVfs7hOAy4H/F8ybAax398OB/wKODbY7GPgmcJqnOgRbBHx5b38MiZ/OLlNFCt1yYIyZ/RR4CPhTJ2UOJXWAeTQ4MS8i9Rh/uy476gI+GfTKeBhwmbtvC+Z/wMy+CpSSGqjlFeDBYFl7p3+LgdHB55OAGwDc/WUzaz+AHA8cDvwtiK0EeDrzVxb5FyV+6Y1eASaHKLeDXa9q+wK4+3ozO4rUoCmXkhpY48IO6xrwirtP7GLbWzLs9253vyyolvmTmT0AbAB+RqpPpDfN7Or2eALbg/dWuv9/aaQGe/lUN+VEOqWqHumNHgf6pA90YWY17XXjaVYA48wsEdxgnRCUHQwk3P1eUlUm7eMNN5IaVhPgH8AQM5sYrJO0PRwExt0XkeqE7Ev8K8mvC+rjwxy4/kbqoISZHQ78ezB/IXCimVUFy/qb2SF7EpvEm874pddxdzezjwE/NrOvAdtIJfnLOxT9G/AG8CqpoTKXBPNHAL80s/YTn68H778CZpvZVmAiqeT8EzMbQOr/yo9JXW3siR8E+/0eqf7mXyY1etdzIdb9GXC7mb0KvBbse6O711uqv/3fmlmfoOw3SfVeKdIt9c4pkqfMrAhIuvs2MxtLqivoQz017qzIXtMZv0j+KgWesNTIbwbMUNKXnqAzfhGRmNHNXRGRmFHiFxGJGSV+EZGYUeIXEYkZJX4RkZj5/zopsEsuADiCAAAAAElFTkSuQmCC\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WsFvVEv3IGDP"
      },
      "source": [
        ""
      ],
      "execution_count": None,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6wJbacIwIGDP"
      },
      "source": [
        ""
      ],
      "execution_count": None,
      "outputs": []
    }
  ]
}
