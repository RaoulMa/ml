# len method takes two different datatypes
print(len("geeks"))
print(len([10, 20, 30]))

# define a method with different number of arguments
def add(x, y, z=0):
    return x + y +z
print(add(1,2,3))
print(add(1,2))

# calling a method belonging to different classes
class Class1:
    def func(self):
        print("Class1")

class Class2:
    def func(self):
        print("Class2")

objects = [Class1(), Class2()]

for obj in objects:
    obj.func()






