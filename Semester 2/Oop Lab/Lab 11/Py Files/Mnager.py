class Manager(Employee):
    def __init__(self, name, age, salary, department):
        super().__init__(name, age, salary)
        self.__department = department

    def get_dept(self):
        return self.__department

    def set_dept(self, dept):
        self.__department = department