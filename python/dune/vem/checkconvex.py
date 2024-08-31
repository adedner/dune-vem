def crossProduct(A):
    x1 = (A[1][0] - A[0][0])
    y1 = (A[1][1] - A[0][1])
    x2 = (A[2][0] - A[0][0])
    y2 = (A[2][1] - A[0][1])
    return (x1*y2 - y1*x2)

def checkConvex(points):
    N = len(points)
    prev = 0
    curr = 0
    for i in range(N):
        temp = [points[i], points[(i + 1) % N], points[(i + 2) % N]]
        curr = crossProduct(temp)
        if (curr != 0):
            if (curr * prev < 0):
                return False
            else:
                prev = curr
    return True
