from exp2 import *
import matplotlib.pyplot as plt

class Test():
    def task4_1(self):
        print('###### Testing Task 4.1 ######')
        a = Interpolation()
        a.plotBinomial()
        a.plotParabola()
        a.plotLn()
        a.plotLn(finer=True)

    def task4_2(self):
        print('###### Testing Task 4.2 ######')
        a = Interpolation()
        a.plotGradBi()
        a.plotGradPa()
        a.plotGradLn()

    def task5_2(self):
        print('###### Testing Task 5.2 ######')
        b = Integration()
        #b.exactIntg(1, 3, d=1)
        b.intg_polynomial_order(1, 3, 1)
        b.quadrature(1, 3, n=2, d=1)
        b.quadrature(1, 3, n=3, d=1)
        b.quadrature(1, 3, n=4, d=1)

        #b.exactIntg(1, 3, d=2)
        b.intg_polynomial_order(1, 3, 2)
        b.quadrature(1, 3, n=2, d=2)
        b.quadrature(1, 3, n=3, d=2)
        b.quadrature(1, 3, n=4, d=2)

        #b.exactIntg(1, 3, d=3)
        b.intg_polynomial_order(1, 3, 3)
        b.quadrature(1, 3, n=2, d=3)
        b.quadrature(1, 3, n=3, d=3)
        b.quadrature(1, 3, n=4, d=3)

        #b.exactIntg(1, 3, d=4)
        b.intg_polynomial_order(1, 3, 4)
        b.quadrature(1, 3, n=2, d=4)
        b.quadrature(1, 3, n=3, d=4)
        b.quadrature(1, 3, n=4, d=4)

        #b.exactIntg(1, 3, d=5)
        b.intg_polynomial_order(1, 3, 5)
        b.quadrature(1, 3, n=2, d=5)
        b.quadrature(1, 3, n=3, d=5)
        b.quadrature(1, 3, n=4, d=5)

        #b.exactIntg(1, 3, d=6)
        b.intg_polynomial_order(1, 3, 6)
        b.quadrature(1, 3, n=2, d=6)
        b.quadrature(1, 3, n=3, d=6)
        b.quadrature(1, 3, n=4, d=6)

        #b.exactIntg(1, 3, d=7)
        b.intg_polynomial_order(1, 3, 7)
        b.quadrature(1, 3, n=2, d=7)
        b.quadrature(1, 3, n=3, d=7)
        b.quadrature(1, 3, n=4, d=7)

    def task5_3(self):
        print('###### Testing Task 5.3 ######')
        LinearSys().oneElement()

    def task7_2(self):
        print('###### Testing Task 7.2 ######')
        print('1. Testing spring system with one Dirichlet BC on the first node:')
        r1 = LinearSys().solveLinear(N=4, u1=0, F=12, c=1)
        r2 = LinearSys().solveLinear(N=4, u1=1.6, F=12, c=1)
        r3 = LinearSys().solveLinear(N=4, u1=3.2, F=12, c=1)
        nodes = np.linspace(1, 4, 4)
        plt.figure()
        plt.plot(nodes, r1, 'ro-', label='u1=0 mm')
        plt.plot(nodes, r2, 'bo-', label='u1=1.6 mm')
        plt.plot(nodes, r3, 'go-', label='u1=3.2 mm')
        plt.xlabel('nodes')
        plt.ylabel('displacement (mm)')
        plt.legend(prop={'size': 15})
        plt.xticks(nodes, fontsize=15)
        plt.yticks(fontsize=15)
        plt.show()

        print('\n2. Testing spring system with Dirichlet BC at both sides:')
        rr1 = LinearSys().solveLinear(N=4, u1=-0.6, uend=1.6, F=12, c=1)
        rr2 = LinearSys().solveLinear(N=4, u1=0, uend=1.6, F=12, c=1)
        rr3 = LinearSys().solveLinear(N=4, u1=0.6, uend=0.6, F=12, c=1)
        nodes = np.linspace(1, 4, 4)
        plt.figure()
        plt.plot(nodes, rr1, 'ro-', label='u1=-0.6, u4=1.6')
        plt.plot(nodes, rr2, 'bo-', label='u1=0, u4=1.6')
        plt.plot(nodes, rr3, 'go-', label='u1=0.6, u4=0.6')
        plt.xlabel('nodes')
        plt.ylabel('displacement (mm)')
        plt.xticks(nodes, fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(prop={'size': 15})
        plt.show()

    def task8_1(self):
        print('###### Testing Task 8.1 ######\n')

        LinearSys().postProcess()



if __name__ == '__main__':
    # Test().task4_1()
    # Test().task4_2()
    # Test().task5_2()
    # Test().task5_3()
    Test().task7_2()
    # Test().task8_1()