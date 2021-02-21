import PyPDF2
import pikepdf
import time
import os


def pdf_Crack(startFile, endFile):
    """
    去除没有打开口令的pdf加密。
    :param startFile: 要破解的pdf文件
    :param endFile: 生成的pdf文件
    :return:生成的pdf文件
    """
    with pikepdf.open(startFile) as pdf:
        pdf.save(endFile)
        return endFile


def batch_pdf_Crack(folderPath):
    """
    将一个文件夹中的文件全部解密，在main.py目录下保存为原名字。
    支持带中文的文件名字。
    支持带中文的路径。
    :param filePath: 要解密的文件夹
    :param savePath: 文件保存的文件夹
    :return:
    """
    filelist = os.listdir(folderPath)
    # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    for file in filelist:
        if file.endswith(".pdf") and ("~$" not in file):
            filePath = folderPath + "\\" + file
            with pikepdf.open(filePath) as pdf:
                pdf.save(file)



if __name__ == "__main__":
    batch_pdf_Crack('C:\\Users\\15594\\Desktop\\文件')
