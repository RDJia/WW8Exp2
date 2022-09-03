import numpy as np
import matplotlib.pyplot as plt
#import exp1
import scipy.linalg

class func():
    '''
    3 functions of choice.
    '''
    def f_Binomial(self, x):
        '''
        Function 1: binomial
        :return fx: function value(s) of given point(s)
        '''
        fx = 2 * np.square(x) + 1
        return fx
    def f_Parabola(self, x):
        '''
        Function 2: Parabola
        :return fx: function value(s) of given point(s)
        '''
        fx = -1 * np.square(x) + 5
        return fx
    def f_ln(self, x):
        '''
        Function 3: log_e
        :return fx: function value(s) of given point(s)
        '''
        fx = np.log(x)
        return fx

    def f_polynomial_order(self, x, order=3):
        '''
        polynomial function n-th order.
        :param order: int, order of polynomial
        :return fx: function value(s) of given point(s)
        '''
        fx = 2 * np.power(x, order) + 1
        return fx

class Interpolation():
    def __init__(self):
        self.x = np.array([1,2,3,4,5,6,7])
        self.y_bi = func().f_Binomial(self.x)
        self.y_pa = func().f_Parabola(self.x)
        self.y_ln = func().f_ln(self.x)

    def evaluationPoints(self,x1, x2, N = 2, exclude=False):
        '''
        Create evenly distributed evaluation points.
        :param x1:  float/int, lower limit
        :param x2:  float/int, upper limit
        :param N:   int, Number of evaluation points
        :param exclude:     boolean, =True to exclude original points
        :return random_points:
                    list, x coordinates for evaluation points
        '''
        points = np.linspace(x1, x2, N+2)
        if exclude == True:
            points = np.delete(points, [0, -1])

        return points

    def lagrange(self, x1, x2, y1, y2, xpoints):
        '''
        Create Lagrange function of 1 element (2 points)
        Approximate the values using random created points.
        :param x1, x2:  float/int, x coordinates of two points
        :param y1, y2:  float/int, y coordinates of two points
        :param xpoints: list/array, random evaluation points
        :return y_lagr: list/array, y values for evaluation points
        '''
        x1_full, x2_full = np.full(len(xpoints), x1), np.full(len(xpoints), x2)
        y_lagr = y1 * (xpoints - x2_full)/(x1_full - x2_full) + y2 * (xpoints - x1_full)/(x2_full - x1_full)

        return y_lagr

    def gradient(self, x1, x2, y1, y2):
        '''
        Calculate finite gradient for an element.
        :param x1, x2:  float/int, x coordinates of two points
        :param y1, y2:  float/int, y coordinates of two points
        :return grad: float, gradient of this element
        '''
        grad = (y2-y1) / (x2-x1)
        return grad

    def plotBinomial(self):
        '''
        Draw binomial curve and plot the interpolation points.
        '''
        plt.figure()

        # plot each element
        for element in range(np.shape(self.x)[0] - 1):
            x1, x2 = self.x[element], self.x[element+1]         # x coordinates of two points
            y1, y2 = self.y_bi[element], self.y_bi[element+1]   # y coordinates of two points

            x_points = self.evaluationPoints(x1, x2)            # evaluation points
            y_lagr = self.lagrange(x1, x2, y1, y2, x_points)    # generate y values by Lagrange

            # for labels of plot
            if element == 0:
                lbl_f = 'original curve'
                lbl_p = 'original points'
                lbl_e = 'evaluation points'
            else:
                lbl_f, lbl_p, lbl_e = None, None, None

            # plot the original function curve
            x_for_func = np.linspace(self.x[element], self.x[element+1], 50)
            y_for_func = func().f_Binomial(x_for_func)
            plt.plot(x_for_func, y_for_func, c='black', label=lbl_f)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            # plot original points
            plt.scatter([self.x[element], self.x[element] + 1], [self.y_bi[element], self.y_bi[element + 1]], c='red',
                        marker='o', label=lbl_p, s=100)
            # plot evaluation points
            plt.scatter(x_points, y_lagr, c='blue', marker='x', label=lbl_e, s=100)

        plt.legend(prop={'size': 15})
        plt.show()

    def plotParabola(self):
        '''
        Draw parabola curve and plot the interpolation points.
        '''
        plt.figure()

        for element in range(np.shape(self.x)[0] - 1):
            x1, x2 = self.x[element], self.x[element + 1]
            y1, y2 = self.y_pa[element], self.y_pa[element + 1]

            x_points = self.evaluationPoints(x1, x2)
            y_lagr = self.lagrange(x1, x2, y1, y2, x_points)

            if element == 0:
                lbl_f = 'original curve'
                lbl_p = 'original points'
                lbl_e = 'evaluation points'
            else:
                lbl_f, lbl_p, lbl_e = None, None, None

            x_for_func = np.linspace(self.x[element], self.x[element + 1], 50)
            y_for_func = func().f_Parabola(x_for_func)
            plt.plot(x_for_func, y_for_func, c='black', label=lbl_f)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

            plt.scatter([self.x[element], self.x[element] + 1], [self.y_pa[element], self.y_pa[element + 1]], c='red',
                        marker='o', label=lbl_p, s=100)
            plt.scatter(x_points, y_lagr, c='blue', marker='x', label=lbl_e, s=100)

        plt.legend(prop={'size': 15})
        plt.show()

    def plotLn(self, finer=False):
        '''
        Draw natural log curve and plot the interpolation points.
        :param finer: boolean, True: element length = 0.5, otherwise 1
        '''
        plt.figure()
        if finer == True:
            x_finer = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
        else:
            x_finer = self.x
        y_ln_finer = func().f_ln(x_finer)

        for element in range(np.shape(x_finer)[0]-1):
            x1, x2 = x_finer[element], x_finer[element+1]
            y1, y2 = y_ln_finer[element], y_ln_finer[element+1]

            x_points = self.evaluationPoints(x1, x2)
            y_lagr = self.lagrange(x1, x2, y1, y2, x_points)

            if element == 0:
                lbl_f = 'original curve'
                lbl_p = 'original points'
                lbl_e = 'evaluation points'
            else:
                lbl_f, lbl_p, lbl_e = None, None, None

            x_for_func = np.linspace(x_finer[element], x_finer[element+1], 50)
            y_for_func = func().f_ln(x_for_func)
            plt.plot(x_for_func, y_for_func, c='black', label=lbl_f)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

            plt.scatter([x1, x2], [y1, y2], c='red',
                        marker='o', label=lbl_e, s=100)
            plt.scatter(x_points, y_lagr, c='blue', marker='x', label=lbl_p, s=100)

        plt.legend(prop={'size': 15})
        plt.show()

    def plotGradBi(self):
        '''
        Plot finite gradient of binomial function, with evaluation points.
        '''
        plt.figure()
        last_grad = 0

        for element in range(np.shape(self.x)[0] - 1):
            x1, x2 = self.x[element], self.x[element+1]
            y1, y2 = self.y_bi[element], self.y_bi[element+1]

            evalPoint = self.evaluationPoints(x1, x2, N=1, exclude=True)
            grad = self.gradient(x1, x2, y1, y2)

            x_for_func = np.linspace(self.x[element], self.x[element+1], 50)
            y_for_func = func().f_Binomial(x_for_func)

            if element == 0:
                plt.plot(x_for_func, y_for_func, c='black', label='binomial curve')
                plt.plot([x1,x2], [grad, grad], c='blue', label='gradient')
                plt.plot([x1,x1], [grad, grad], c='blue')
            else:
                plt.plot(x_for_func, y_for_func, c='black')
                plt.plot([x1,x2], [grad, grad], c='blue')
                plt.plot([x1, x1], [last_grad, grad], c='blue')

            if element == 0:
                lbl_e = 'evaluation points'
            else:
                lbl_e = None

            plt.scatter(evalPoint, grad, c='red', marker='o', label=lbl_e)
            last_grad = grad

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(prop={'size': 15})
        plt.show()

    def plotGradPa(self):
        '''
        Plot gradient of parabola function, with evaluation points.
        '''
        plt.figure()
        last_grad = 0

        for element in range(np.shape(self.x)[0] - 1):
            x1, x2 = self.x[element], self.x[element+1]
            y1, y2 = self.y_pa[element], self.y_pa[element+1]

            evalPoint = self.evaluationPoints(x1, x2, N=1, exclude=True)
            grad = self.gradient(x1, x2, y1, y2)

            x_for_func = np.linspace(self.x[element], self.x[element+1], 50)
            y_for_func = func().f_Parabola(x_for_func)

            if element == 0:
                plt.plot(x_for_func, y_for_func, c='black', label='parabola curve')
                plt.plot([x1,x2], [grad, grad], c='blue', label='gradient')
                plt.plot([x1,x1], [grad, grad], c='blue')
            else:
                plt.plot(x_for_func, y_for_func, c='black')
                plt.plot([x1,x2], [grad, grad], c='blue')
                plt.plot([x1, x1], [last_grad, grad], c='blue')

            if element == 0:
                lbl_e = 'evaluation points'
            else:
                lbl_e = None

            plt.scatter(evalPoint, grad, c='red', marker='o', label=lbl_e)
            last_grad = grad

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(prop={'size': 15})
        plt.show()

    def plotGradLn(self, finer=False):
        '''
        Plot gradient of logn function, with evaluation points.
        :param finer: boolean, True: element length = 0.5, otherwise 1
        '''
        plt.figure()
        last_grad = 0

        if finer == True:
            x_finer = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
        else:
            x_finer = self.x
        y_ln_finer = func().f_ln(x_finer)

        for element in range(np.shape(x_finer)[0] - 1):
            x1, x2 = x_finer[element], x_finer[element+1]
            y1, y2 = y_ln_finer[element], y_ln_finer[element+1]

            evalPoint = self.evaluationPoints(x1, x2, N=1, exclude=True)
            grad = self.gradient(x1, x2, y1, y2)

            x_for_func = np.linspace(x1, x2, 50)
            y_for_func = func().f_ln(x_for_func)

            if element == 0:
                plt.plot(x_for_func, y_for_func, c='black', label='ln curve')
                plt.plot([x1,x2], [grad, grad], c='blue', label='gradient')
                plt.plot([x1,x1], [grad, grad], c='blue')
            else:
                plt.plot(x_for_func, y_for_func, c='black')
                plt.plot([x1,x2], [grad, grad], c='blue')
                plt.plot([x1, x1], [last_grad, grad], c='blue')

            if element == 0:
                lbl_e = 'evaluation points'
            else:
                lbl_e = None

            plt.scatter(evalPoint, grad, c='red', marker='o', label=lbl_e)
            last_grad = grad

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(prop={'size': 15})
        plt.show()

class Integration():
    def quadrature(self, x1, x2, n, d):
        '''
        Do quadrature integration.
        :param x1:   float,  starting point of integral
        :param x2:   float,  ending point of integral
        :param n:    int,    degree of Newton-Cotes formula
        :param d:    int,    degree of polynomial
        :return quad_integration:   float, result of quadrature integration
        '''
        quad_integration = 0

        para_zeta, para_lambda = self.NewtonCotes(x1, x2, n)    # get lambda & zeta
        f_zeta = func().f_polynomial_order(para_zeta, order=d)  # get f(zeta)

        for i in range(n):
            quad_integration += para_lambda[i] * f_zeta[i]      # do summation
        quad_integration = quad_integration * (x2 - x1)

        print(f'integral of n={n}, d={d}:  ', quad_integration)
        return quad_integration

    def NewtonCotes(self, x1, x2, n):
        '''
        Create lambda and zeta given degree n.
        :param x1:  float,  starting point of integral
        :param x2:  float,  ending point of integral
        :param n:   int, degree of Newton-Cotes formula
        :return para_zeta:  ndarray, zeta values
        :return para_lambda:  ndarray, lambda values
        '''
        if n == 2:
            para_zeta = [x1, x2]
            para_lambda = [0.5, 0.5]
        elif n == 3:
            para_zeta = [x1, 0.5*(x1+x2), x2]
            para_lambda = [1/6, 4/6, 1/6]
        elif n == 4:
            para_zeta = [x1, (2*x1+x2)/3, (x1+2*x2)/3, x2]
            para_lambda = [1/8, 3/8, 3/8, 1/8]

        return para_zeta, para_lambda
    
    def exactIntg(self, x1, x2, d, n=10000):
        '''
        Approximate real integral value by splitting the area into many small elements.
        :param x1:  float,  starting point of integral
        :param x2:  float,  ending point of integral
        :param n:   int,    degree of Newton-Cotes formula
        :param d:   int,    degree of polynomial
        :return area:   float, approximated integral value
        '''
        x_points = np.linspace(x1, x2, n)
        y_points = func().f_polynomial_order(x_points, order=d)

        area = 0
        for i in range(n):
            if i == n-1:
                area += (x2 - x_points[i]) * y_points[i]
            else:
                area += (x_points[i+1] - x_points[i]) * y_points[i]

        print(f'=====real area of d={d}:', area, '=====')
        return area

class LinearSys():
    def oneElement(self, x1=0, x2=50, E=210000, A=25, l=50, f_b=0, f_s=5):
        '''
        Special case for Task 5_3: one element with BC on first node (x=0, u=0).
        Test 1 element given following values.
        :param x1, x2:  float,  starting/ending position (mm)
        :param E:  float,  E-Modul (N/mm^2)
        :param A:  float,  cross section area (mm^2)
        :param l:  float,  element length (mm)
        :param f_b, f_s:  float,  force on starting/ending point (N)
        '''

        Ke = E * A * 2 * l / (x1-x2)**2
        u2 = (f_b+f_s) / Ke

        print('displacement of ending point ≈', np.round(u2, 7))

    def solveLinear(self, N = 6, u1 = 0, uend = None, F = -5, c = 1, to_print = True):
        '''
        Create permutation matrix.
        :param N:       int, number of elements
        :param u1:      float, displacement on first node (Dirichlet BC)
        :param uend:    float, displacement on last node (Dirichlet BC)
                        = None if no Dirichlet BC on it
        :param F:       float, force applied on the last node
        :param c:       float, multiplier of global stiffness matrix
        :param to_print:    boolean, whether to print the parameters
        :return d:      ndarray, solution, displacement matrix
        '''

        if to_print == True:
            print('==============================')
            print('Testing: N =', N, ', u1 =', u1, ', uend =', uend, ', F =', F, ', c =', c )

        if uend is None:        # P for case with only 1 Dirichlet BC at node 1.
            P = np.zeros([N-1, N])
            for row in range(N-1):
                for col in range(N):
                    if row + 1 == col:
                        P[row,col] = 1
        else:                   # P for Dirichlet BCs at both sides
            P = np.zeros([N - 2, N])
            for row in range(N-2):
                for col in range(N):
                    if row + 1 == col:
                        P[row,col] = 1

        # Create K.
        K = np.zeros([N, N])
        for row in range(N):
            for col in range(N):
                if row == col and row != 0 and row != N-1:
                    K[row, col] = 2
                    K[row, col-1], K[row, col+1] = -1, -1
        K[0, 0], K[N - 1, N - 1] = 1, 1
        K[0, 1], K[N - 1, N - 2] = -1, -1
        K = c * K

        d_dirichlet = np.zeros([N,1])
        d_dirichlet[0] = u1
        if uend is not None:
            d_dirichlet[-1] = uend

        # reduced right hand side
        f = np.zeros([N, 1])
        f[-1] = F

        fr0 = f - np.dot(K, d_dirichlet)
        fr = np.dot(P, fr0)

        Kr = np.dot(np.dot(P, K), P.T)

        dr = scipy.linalg.solve(c*Kr, fr)

        d = np.dot(P.T, dr) + d_dirichlet

        if to_print == True:
            print('========= P =========\n', P)
            print('=== d ===\n', d)
            print('=== K ===\n', K)
            print('=== d_dirichlet ===\n', d_dirichlet)
            print('=== f ===\n', f)
            print('=== fr ===\n', fr)
            print('=== Kr ===\n', Kr)
            print('=== dr ===\n', dr)

        return d

    def postProcess(self):
        '''
        Post process:
        Calculate and plot stress tensors under different BCs.
        '''
        d_1 = []    # displacements
        BC_1 = np.array([[0, None], [15, None], [30, None], [15,15], [15,45], [30,45]])

        # BCs on first node (u1)
        d_1.append(self.solveLinear(N=6, u1=BC_1[0][0], F=12, c=1, to_print=False))
        d_1.append(self.solveLinear(N=6, u1=BC_1[1][0], F=12, c=1, to_print=False))
        d_1.append(self.solveLinear(N=6, u1=BC_1[2][0], F=12, c=1, to_print=False))
        # BCs on first and last nodes (u1 and uend)
        d_1.append(self.solveLinear(N=6, u1=BC_1[3][0], uend=BC_1[3][1], F=12, c=1, to_print=False))
        d_1.append(self.solveLinear(N=6, u1=BC_1[4][0], uend=BC_1[4][1], F=12, c=1, to_print=False))
        d_1.append(self.solveLinear(N=6, u1=BC_1[5][0], uend=BC_1[5][1], F=12, c=1, to_print=False))

        E = 210000  # E-Modul

        stress_fields = []  # list of all stress fields
        x_points = np.linspace(1, 6, 6) # nodes

        # Calculate gradient for on displacement results based on different BCs.
        for BC in range(6):

            grad_1 = np.zeros(6)
            for i in range(5):
                grad_1[i + 1] = d_1[BC][i + 1] - d_1[BC][i][0]
            grad_1[0] = grad_1[1]

            this_stress_field = [] # stress field under this particular BC
            
            for node in range(6):
                print('grad_1[node]:', grad_1[node])
                print('d_1[BC][node][0]:', d_1[BC][node][0])
                this_stress_field.append(E * grad_1[node]* d_1[BC][node][0])

            stress_fields.append(this_stress_field)

            print(f'{BC + 1}. result of u1 = {BC_1[BC][0]} mm, uend = {BC_1[BC][1]} mm \n')
            print('*** displacement ***: \n', d_1[BC].T)
            print('***  gradient ***: \n', grad_1)
            print('*** stress field ***: \n', this_stress_field, '\n')

            # Plot results
            plt.figure(figsize=(10, 8), dpi=80)

            ax_sf = plt.subplot(122)
            ax_sf.plot(this_stress_field,'g-o')
            ax_sf.set_title('stress field (Pa)')
            ax_sf.xaxis.tick_top()

            ax_d = plt.subplot(221)
            ax_d.plot(x_points, d_1[BC], '-o')
            ax_d.set_ylim(-1, 90)
            ax_d.set_title(f'displacement  of u1 = {BC_1[BC][0]} mm, uend = {BC_1[BC][1]} mm')
            ax_d.set_ylabel('displacement (mm)')

            ax_g = plt.subplot(223)
            ax_g.plot(x_points, grad_1, 'r-o')
            #ax_g.set_ylim(0, 0.08)
            ax_g.set_title(f'gradient of u1 = {BC_1[BC][0]} mm, uend = {BC_1[BC][1]} mm')
            ax_g.set_ylabel('gradient')

            plt.suptitle(f'u1 = {BC_1[BC][0]} mm, uend = {BC_1[BC][1]} mm\n N = 6, F = 12 N, c = 1')
            plt.show()

        # plot some stress fields in a figure for comparison
        plt.figure()
        ax = plt.plot(x_points, stress_fields[0], '-o', label='u1 = 0 mm')
        plt.plot(x_points, stress_fields[1], '-o', label='u1 = 15 mm')
        plt.plot(x_points, stress_fields[3], '-o', label='u1 = u6 = 15 mm')
        plt.plot(x_points, stress_fields[4], '-o', label='u1 = 15mm, u6 = 45mm')
        plt.xlabel('nodes')
        plt.ylabel('stress (Pa)')
        plt.legend()
        plt.show()
            



