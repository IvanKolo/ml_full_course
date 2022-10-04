x = 1
y = 2
x += 5
l = [x, y]

a = [1]
b = [2]
s = [a, b]
a.append(5)

print(l)
print(s)


def f(x, s=set()):
    s.add(x)
    print(s)



f(7)
f(6, {4, 5})
f(2)
