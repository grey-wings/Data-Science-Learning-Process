import PyPDF2
import pikepdf
import time
import os
from glob import glob


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


# def split_PDF_pages(filePath, savePath, pageEnd, pageBeg):
#     """
#     将pdf的第pageBeg到第pageEnd页截取出来,保存至savePath.
#     :param filePath:
#     :param pageBeg:
#     :param pageEnd:
#     :return:
#     """
#     # 未成功
#     with pikepdf.open(filePath) as pdf:
#         dst = pikepdf.new()
#         for n, page in enumerate(pdf.pages[pageBeg-1:pageEnd]):
#             dst.pages.append(page)
#         dst.save(savePath)


def deletePages(filePath, savePath, begin, end):
    """
    删除pdf的第begin页到第end页，第end页也被删除，计数从1开始。
    :param filePath: 要删除页面的文件
    :param begin: 开始删除的页面（第1页为1）
    :param end: ……
    :return:
    """
    with pikepdf.open(filePath) as pdf:
        del pdf.pages[begin - 1:end]
        pdf.save(savePath)


def PDF_Merge(filePath1, filePath2, savePath):
    """
    合并两个pdf文件
    :param filePath1:
    :param filePath2:
    :param savePath: 保存路径和文件名
    :return:
    """
    pdf = pikepdf.new()
    with pikepdf.open(filePath1) as pdf1:
        pdf.pages.extend(pdf1)
        with pikepdf.open(filePath2) as pdf2:
            pdf.pages.extend(pdf2)


if __name__ == "__main__":
    pass
