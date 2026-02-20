class A:
    pass

class B:
    pass

class_type = A

match class_type:
    case A:
        print("A")
    case B:
        print("B")
