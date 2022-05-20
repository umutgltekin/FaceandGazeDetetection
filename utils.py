

def delete_ops():
    file_name = 'requirements.txt'

    text = ""
    with open(file_name, 'r') as file:
        for f in file:
            text += f.split("@")[0]

    file2_name = "req.txt"
    with open(file2_name, "a+") as file:
        file.write(text)


if __name__ == '__main__':
    delete_ops()
