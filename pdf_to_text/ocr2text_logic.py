import os
import shutil
import errno
import subprocess
import tempfile
from tempfile import mkdtemp
from PIL import Image
import pytesseract
#from pdf2image import convert_from_path, convert_from_bytes
import sys


def update_progress(progress):
    barLength = 10  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format(
        "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def run(args):
        # run a subprocess and put the stdout and stderr on the pipe object
        try:
            pipe = subprocess.Popen(
                args,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except OSError as e:
            if e.errno == errno.ENOENT:
                # File not found.
                # This is equivalent to getting exitcode 127 from sh
                raise Exception.ShellError(
                    ' '.join(args), 127, '', '',
                )

        # pipe.wait() ends up hanging on large files. using
        # pipe.communicate appears to avoid this issue
        stdout, stderr = pipe.communicate()

        # if pipe is busted, raise an error (unlike Fabric)
        if pipe.returncode != 0:
            raise Exception.ShellError(
                ' '.join(args), pipe.returncode, stdout, stderr,
            )

        return stdout, stderr


def extract_tesseract(filename, temp_fp):
    def append(page_nr):
        print(f'{page_nr} pages of file {filename[-15:]} converted')
        with open(temp_fp, 'a', encoding='utf-8') as f_out:  # append if doesn't exist
            f_out.write(''.join(contents))

    temp_dir = mkdtemp()
    base = os.path.join(temp_dir, 'conv')

    contents = []
    try:
        stdout, _ = run(['pdftoppm', filename, base])

        for k, page in enumerate(sorted(os.listdir(temp_dir)), 1):
            page_path = os.path.join(temp_dir, page)
            page_content = pytesseract.image_to_string(Image.open(page_path))
            contents.append(page_content)
            if k % 20 == 0:
                append(k)
                contents = []

        append(k)
        return temp_fp
    finally:
        shutil.rmtree(temp_dir)


def convert_recursive(source, destination, count):
    pdfCounter = 0
    for dirpath, dirnames, files in os.walk(source):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdfCounter += 1
    print()
    ''' Helper function for looping through files recursively '''
    for dirpath, dirnames, files in os.walk(source):
        for name in files:
            filename, file_extension = os.path.splitext(name)
            if (file_extension.lower() != '.pdf'):
                continue
            relative_directory = os.path.relpath(dirpath, source)
            source_path = os.path.join(dirpath, name)
            output_directory = os.path.join(destination, relative_directory)
            output_filename = os.path.join(output_directory, filename + '.txt')
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            temp_filename = os.path.join(output_directory, filename + '-temp.txt')
            count = convert(source_path, output_filename, count, pdfCounter, temp_filename)
    return count


def convert(sourcefile, destination_file, count, pdfCounter, temp_filename):
    extract_tesseract(sourcefile, temp_filename)
    if os.path.exists(temp_filename):
        os.rename(temp_filename, destination_file)
    print()
    print('Converted ' + sourcefile)
    count += 1
    update_progress(count / pdfCounter)
    return count
