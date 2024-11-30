filename =  "Lab11.csv"

def load_employees():
    employees= []
    try:
        with open(filename, mode = "r") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                if row["dept"]:
                    employees.append(Manager(row["name"], int(row["age"]), float(row["salary"]), row["dept"]))

                elif row["hours_worked"]:
                    employees.append(Worker(row["name"],int(row["age"]), float(row["salary"]),int(row["hours_worked"])))


    except FileNotFoundError:
        pass

    return employees


def save_employees(employees):
    with open(filename, mode = "w", newline= "") as file:
        fieldnames = ["name", "age", "salary", "dept", "hours_worked"]
        csv_writer = csv.DictWriter(file, fieldnames = fieldnames)
        csv_writer.writeheader()
        for emp in employees:
            if isinstance(emp, Manager):
                csv_writer.writerow({
                    "name" : emp.get_name(),
                    "age": emp.get_age(),
                    "salary": emp.get_salary(),
                    "dept" : emp.get_dept(),
                    "hours_worked": ""
                })
            elif isinstance(emp, Worker):
                csv_writer.writerow({
                    "name": emp.get_name(),
                    "age" : emp.get_age(),
                    "salary" : emp.get_salary(),
                    "dept" : "",
                    "hours_worked" : emp.get_hours_worked()
                })


def add_employee():
    employee_type = input("Enter employee type(Manager/Worker): ").strip().lower()
    name = input("Enter name: ")
    age = int(input("Enter age: "))
    salary = float(input("Enter salary: "))

    if employee_type == "manager":
        department = input("Enter department: ")
        new_employee = Manager(name, age, salary, department)

    elif employee_type == "worker":
        hours_worked = int(input("Enter Hours Worked: "))
        new_employee = Worker(name, age, salary, hours_worked)

    else:
        print("Invalid employee type.")
        return None

    employees = load_employees()
    employees.append(new_employee)
    save_employees(employees)
    print("Employee is added successfully.")


def display_employees():
    employees = load_employees()
    if not employees:
        print("No employee found.")
        return

    for emp in employees:
        print("Name:", emp.get_name())
        print("Age:", emp.get_age())
        print("Salary:", emp.get_salary())
        if isinstance(emp, Manager):
            print("Dept:", emp.get_dept())

        elif isinstance(emp, Worker):
            print("Hours Worked:", emp.get_hours_worked())

        print("-" * 30)


def update_employee():
    name = input("Enter the name of the employee to update:")
    employees = load_employees()
    for emp in employees:
        if emp.get_name().lower() == name.lower():
            new_name = input("Enter new name: ")
            if new_name:
                emp.set_name(new_name)

            new_age = input("Enter new age: ")
            if new_age:
                emp.set_age(int(new_age))

            new_salary = input("Enter new salary:")
            if new_salary:
                emp.set_salary(float(new_salary))

            if isinstance(emp, Manager):
                new_dept = input("Enter new departmant:")
                if new_dept:
                    emp.set_dept(new_dept)

            elif isinstance(emp, Worker):
                new_hours = input("Enter new hours worked: ")
                if new_hours:
                    emp.set_hours_worked(int(new_hours))

            save_employees(employees)
            print("Employee updated successfully.")
            return

    print("Employee not found.")


def delete_employee():
    name = input("Enter name of employee for delete: ")
    employees = load_employees()
    updated_employees = [emp for emp in employees if emp.get_name().lower() != name.lower()]

    if len(updated_employees) == len(employees):
        print("Employee not found.")

    else: 
        save_employees(updated_employees)
        print("Employees deleted successfully.")


def main():
    while True:
        print("\n Employee Management System")
        print("1. Add Employee")
        print("2. Display All Employee")
        print("3. Update Employee")
        print("4. Delete Employee")
        print("5. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            add_employee()

        elif choice == "2":
            display_employees()

        elif choice == "3":
            update_employee()

        elif choice == "4":
            delete_employee()

        elif choice == "5":
            print("Existing the program...")
            break

        else:
            print("Invalid choice...")

if __name__ == "__main__":
    main()