class Rectangle:
    def __init__(self, height: int, width: int):
        self.h = height
        self.w = width

    def size(self) -> int:
        return self.h * self.w
    
    def CopyNew(self) -> 'Rectangle':
        return Rectangle(self.h, self.w)
    
    def shape(self) -> str:
        return f"({self.h}, {self.w})"  

class Cuboid:
    def __init__(self, length: int, height: int, width: int):
        self.z = length
        self.h = height
        self.w = width

    def size2d(self) -> int:
        return self.h * self.w

    def size3d(self) -> int:
        return self.z * self.h * self.w
    
    def CopyNew(self) -> 'Cuboid':
        return Cuboid(self.z, self.h, self.w)
    
    def shape(self) -> str:
        return f"({self.z}, {self.h}, {self.w})"
    
class Output:
    def __init__(self, ciphertexts: list, size: Cuboid, interval: Rectangle=Rectangle(1, 1), const: int=1):
        self.ciphertexts = ciphertexts  # Ciphertext list
        self.size = size.CopyNew()
        self.interval = interval.CopyNew()
        self.const = const

    def CopyNew(self) -> 'Output':
        return Output(self.ciphertexts, self.size, self.interval, self.const)