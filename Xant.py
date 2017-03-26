# -*- coding: UTF-8 -*-
import random
from math import pow, sqrt
import numpy as np
import pandas as pd
import pylab as pl


class MMAS:
    def __init__(self, antCount=20, q=100, alpha=1, beta=3,
                 rou=0.3, initialph=10, nMax=10000):
        '''
        https://svn-d1.mpi-inf.mpg.de/AG1/MultiCoreLab/papers/StuetzleHoos00%20-%20MMAS.pdf
        '''
        self.AntCount = antCount
        self.Q = q
        self.Alpha = alpha
        self.Beta = beta
        self.Rou = rou
        self.initialPh = initialph
        self.Nmax = nMax
        self.Shortest = float('inf')
        self.AntList = []
        pl.show()

    def ReadCityInfo(self, fileName):
        '''
        http://stackoverflow.com/questions/29281680/numpy-individual-element-access-slower-than-for-lists
        '''
        city_info = pd.read_csv(fileName,
                                sep=' ',
                                skiprows=6, skipfooter=1,
                                engine='python',
                                header=None,
                                names=('N', 'x', 'y'))
        self.CityCount = city_info.shape[0]
        self.CitySet = set()
        self.CityDistance = [
            [0] * self.CityCount for i in range(self.CityCount)]
        self.CityDistanceBeta = np.zeros(
            (self.CityCount, self.CityCount)).tolist()
        self.Pheromone = [
            [self.initialPh] * self.CityCount for i in range(self.CityCount)]
        self.PheromoneDelta = np.zeros(
            (self.CityCount, self.CityCount)).tolist()
        self.BestTour = [None] * self.CityCount

        for row in city_info.index:
            for col in city_info.index:
                if row != col:
                    distance = round(
                        sqrt(pow(city_info.x[row] - city_info.x[col], 2)
                             + pow(city_info.y[row] - city_info.y[col], 2))
                    )
                    self.CityDistance[row][col] = distance
                    self.CityDistanceBeta[row][col] = pow(
                        1.0 / distance, self.Beta)
        self.city_info = city_info

    def PutAnts(self):
        self.AntList.clear()
        for antNum in range(self.AntCount):
            city = random.choice(self.city_info.index)
            ant = ANT(city, self.city_info.index,
                      self.CityDistance, self.Pheromone)
            self.AntList.append(ant)

    def Search(self):
        import time
        for iter in range(self.Nmax):
            start = time.time()
            self.PutAnts()
            tmpLen = float('inf')
            tmpTour = []
            for ant in self.AntList:
                for ttt in range(self.CityCount):
                    ant.MoveToNextCity(self.Alpha, self.Beta)
                ant.two_opt_search()
                ant.UpdatePathLen()
                if ant.CurrLen < tmpLen:
                    self.bestAnt = ant
                    tmpLen = ant.CurrLen
                    tmpTour = ant.TabuCityList
            if tmpLen < self.Shortest:
                self.Shortest = tmpLen
                self.BestTour = tmpTour
            print(iter, "-->", self.Shortest, "-->", self.BestTour)
            # self.bestAnt.two_opt_search()
            self.UpdatePheromoneTrail()
            end = time.time()
            print(end - start)

            pl.clf()
            x = []; y = []
            for city in self.BestTour:
                x.append(self.city_info.x[city])
                y.append(self.city_info.y[city])
            x.append(x[0])
            y.append(y[0])
            pl.plot(x, y)
            pl.scatter(x, y, s=30, c='r')
            pl.pause(0.01)

    def UpdatePheromoneTrail(self):
        ant = self.bestAnt
        pheromo_new = self.Q / ant.CurrLen
        tabu = ant.TabuCityList
        PD = self.PheromoneDelta
        P = self.Pheromone
        citys = self.city_info.index

        for city, nextCity in zip(tabu[:-1], tabu[1:]):
            PD[city][nextCity] = pheromo_new
            PD[nextCity][city] = pheromo_new
        lastCity = tabu[-1]
        firstCity = tabu[0]
        PD[lastCity][firstCity] = pheromo_new
        PD[firstCity][lastCity] = pheromo_new

        for c1 in citys:
            for c2 in citys:
                if c1 != c2:
                    P[c1][c2] = (
                        (1 - self.Rou) * P[c1][c2]
                        + PD[c1][c2]
                    )
                    if P[c1][c2] < 0.001:
                        P[c1][c2] = 0.001
                    if P[c1][c2] > 10:
                        raise(Exception('too big Ph'))
                        P[c1][c2] = 10
                    PD[c1][c2] = 0


class ANT:
    def __init__(self, currCity=0, citys=None, cityDis=None, pheromo=None):
        self.TabuCityList = [currCity, ]
        self.AllowedCitySet = set(citys)
        self.AllowedCitySet.remove(currCity)
        self.CityDistance = cityDis
        self.Pheromone = pheromo
        self.TransferProbabilityList = []
        self.CurrCity = currCity
        self.CurrLen = 0

    def SelectNextCity(self, alpha, beta):
        if self.AllowedCitySet:
            sumProbability = 0
            self.TransferProbabilityList = []
            for city in self.AllowedCitySet:
                sumProbability = sumProbability + (
                    pow(self.Pheromone[self.CurrCity][city], alpha) *
                    pow(1.0 /
                        self.CityDistance[self.CurrCity][city], beta)
                )
                transferProbability = sumProbability
                self.TransferProbabilityList.append(
                    (city, transferProbability))
            threshold = sumProbability * random.random()
            #~print(self.TransferProbabilityList)
            for cityNum, cityProb in self.TransferProbabilityList:
                if threshold <= cityProb:
                    return (cityNum)
        return None

    def MoveToNextCity(self, alpha, beta):
        '''
        对于有0返回值的if语句不能使用if x: ... 判断
        '''
        nextCity = self.SelectNextCity(alpha, beta)
        if nextCity is not None:
            self.CurrCity = nextCity
            self.TabuCityList.append(nextCity)
            self.AllowedCitySet.remove(nextCity)

    def UpdatePathLen(self):
        for city, nextCity in zip(self.TabuCityList[:-1],
                                  self.TabuCityList[1:]):
            self.CurrLen = self.CurrLen + self.CityDistance[city][nextCity]
        #~print(self.TabuCityList)
        lastCity = self.TabuCityList[-1]
        firstCity = self.TabuCityList[0]
        self.CurrLen = self.CurrLen + self.CityDistance[lastCity][firstCity]

    def two_opt_search(self):
        '''
        1-2-3-4, 1-2 + 3-4 > 1-3 + 2-4 则交换
        '''
        cityNum = len(self.TabuCityList)
        for i in range(cityNum):
            for j in range(cityNum - 1, i, -1):
                curCity1 = self.TabuCityList[i]
                preCity1 = self.TabuCityList[(i - 1) % cityNum]
                curCity2 = self.TabuCityList[j]
                nextCity2 = self.TabuCityList[(j + 1) % cityNum]
                CurrLen = self.CityDistance[preCity1][
                    curCity1] + self.CityDistance[curCity2][nextCity2]
                NextLen = self.CityDistance[preCity1][
                    curCity2] + self.CityDistance[curCity1][nextCity2]
                if NextLen < CurrLen:
                    tempList = self.TabuCityList[i:j + 1]
                    self.TabuCityList[i:j + 1] = tempList[::-1]


if __name__ == '__main__':
    aco = MMAS()
    aco.ReadCityInfo('eil51.tsp')
    aco.Search()
