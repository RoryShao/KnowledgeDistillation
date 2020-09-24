import matplotlib.pyplot as plt
import Teacher
import Student

if __name__ == '__main__':
    teacher_model, teacher_history = Teacher.teacher_main()
    # print("teacher_model:")
    # print(teacher_model)
    # print("teacher_history:")
    # print(teacher_history)
    student_kd_model, student_kd_history = Student.student_kd_main(teacher_model)
    # print('student_kd_model:')
    # print(student_kd_model)
    # print('student_kd_history:')
    # print(student_kd_history)
    student_simple_model, student_simple_history = Student.student_main()
    # print("student_simple_model:")
    # print(student_simple_model)
    # print("student_simple_history:")
    # print(student_simple_history)
    # 三个模型的loss和acc分析
    epochs = 10
    x = list(range(1, epochs + 1))
    print(x)

    # teacher_history = [(3.781738744877192, 0.09964253798033959), (5.772407047656948, 0.0969615728328865),
    #  (6.477356066120001, 0.09785522788203753)]
    # student_kd_history = [(2.3090088326980003, 0.10031277926720286), (2.343992449749358, 0.0990840035746202),
    #  (2.475689271819496, 0.10098302055406613)]
    #
    # student_simple_history = [(9.338520394360199, 0.0985254691689008), (11.320208429331434, 0.09874888293118857),
    #  (12.066752388607295, 0.10020107238605898)]

    plt.subplot(2, 1, 1)
    plt.plot(x, [teacher_history[i][1] for i in range(epochs)], label='teacher')
    plt.plot(x, [student_kd_history[i][1] for i in range(epochs)], label='student with KD')
    plt.plot(x, [student_simple_history[i][1] for i in range(epochs)], label='student without KD')

    plt.title('Test accuracy')
    plt.legend()
    # plt.show()

    plt.subplot(2, 1, 2)
    plt.plot(x, [teacher_history[i][0] for i in range(epochs)], label='teacher')
    plt.plot(x, [student_kd_history[i][0] for i in range(epochs)], label='student with KD')
    plt.plot(x, [student_simple_history[i][0] for i in range(epochs)], label='student without KD')

    plt.title('Test Loss')
    plt.legend()
    plt.show()
