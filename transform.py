import xlwt  # .py文件要在和xlwt和xlrd同一层目录，不然报错

wb = xlwt.Workbook(encoding='utf-8')
ws = wb.add_sheet('Sheet1')  # sheet页第一页

wb1 = xlwt.Workbook(encoding='utf-8')
ws1 = wb1.add_sheet('Sheet1')  # sheet页第一页

f = open('m.txt', encoding='gbk')  # .py文件和TestCase.txt同一目录，第一个参数是路径
f1 = open('l.txt', encoding='gbk')  # .py文件和TestCase.txt同一目录，第一个参数是路径

row_excel = 0  # 行

for line in f:
    line = line.strip('\n')  # 去掉换行符
    line = line.split(' ')  # 每一行以"+"分隔

    print(line)  # 测试

    col_excel = 0  # 列
    len_line = len(line)
    for j in range(len_line):
        print(line[j])  # 测试
        ws.write(row_excel, col_excel, line[j])
        col_excel += 1
        wb.save('m.xls')  # 输出在同一目录

    row_excel += 1

f.close

row_excel1 = 0  # 行

for line1 in f1:
    line1 = line1.strip('\n')  # 去掉换行符
    line1 = line1.split(' ')  # 每一行以"+"分隔

    print(line1)  # 测试

    col_excel1 = 0  # 列
    len_line1 = len(line1)
    for j in range(len_line1):
        print(line1[j])  # 测试
        ws1.write(row_excel1, col_excel1, line1[j])
        col_excel1 += 1
        wb1.save('l.xls')  # 输出在同一目录

    row_excel1 += 1

f1.close
