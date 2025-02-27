# Disagreement & Misinformation
This is the repository for the paper ["Disagreement as a way to study misinformation and its effects"](https://arxiv.org/abs/2408.08025)

## Installation
* Install dependencies: `pip install -r requirements.txt`


## Disagreement measurement in letters to the editors 

To replicate the figure in the paper, run the following script:
- Navigate to the `measurement` directory.
- Run `python letters_nyt.py`

We have included the required files in the repository:
- `datasets/yearly_data_from_book.csv`: Number of letters and percentage of conspiracy theories on a yearly basis from the book "American Conspiracy Theories".
- `data_output/book_letter.csv`:  Disagreement analysis of the same letters as in the book.
- `data_output/proquest_letters.csv.zip`: Disagreement analysis of letters to the editor accessed through ProQuest.

To run the disagreement analysis, proceed as follows:

### ProQuest Letters
Because the letters cannot be downloaded from ProQuest, the analysis needs to be run on the [ProQuest server](https://www.proquest.com)
The IDs of the letters are provided in a dataset repository on Harvard Dataverse: [https://doi.org/10.7910/DVN/UBLQPA](https://doi.org/10.7910/DVN/UBLQPA).
On ProQuest, proceed as follows:
- Navigate to the `measurement` directory.
- Run `python letters_proquest.py`

### Book Letters
To access the letters from the book "American Conspiracy Theories", please reach out to the authors of the book.
We have included a simple script to convert the PDFs to text files, see [Converting PDFs to TXT](#Converting-PDFs-to-TXT) below.
Once you have the text files placed in the `datasets/amct_text_files` directory, proceed as follows:
- Navigate to the `measurement` directory.
- Run `python letters_book.py`

## Disagreement measurement in tweets
To replicate the figure on vaccine disagreement and misinformation in the appendix of the paper, run the following script:

- Navigate to the `measurement` directory.
- Run `python tweets_vaccine.py`

We have included the required file in the repository:
- `data_output/tweet_dis_misinfo.csv`: Disagreement and misinformation analysis of the vaccine tweets. 

To generate this file by running the analysis, proceed as follows:
First, the tweets need to be rehydrated. The IDs of the tweets are provided in a dataset repository on Harvard Dataverse: [https://doi.org/10.7910/DVN/UBLQPA](https://doi.org/10.7910/DVN/UBLQPA).
The tweets (text and date) need to placed in the `datasets/tweet_datasets` directory in the parquet format.
Then, proceed as follows:
- Define the file name(s) of the tweet datasets as list named `TWEET_FILES` in `measurment/helpers.py`
- Navigate to the `measurement` directory.
- Run `python tweets_vaccine_analysis.py`

The misinformation uses a fine-tuned BERT model which can be accessed here: [https://huggingface.co/hodeld/Anti-Vax-Misinfo-Detector](https://huggingface.co/hodeld/Anti-Vax-Misinfo-Detector).
We have included a Jupyter notebook to run the training of the model in `misinfo_model/Bert_CovidVaccine_Misinformation.ipynb`


## Converting PDFs to TXT
This simple app is used to convert letters in PDF format to TXT files using OCR.
The code is based on [ocr2text](https://github.com/writecrow/ocr2text/) which uses [Google Tesseract-OCR](https://github.com/tesseract-ocr/tesseract).


### Installing Tesseract
Tesseract needs to be installed separately. For instructions, see the [ocr2text README](https://github.com/writecrow/ocr2text/blob/main/README.md)

### Running the script
- Place the PDF Files in the `datasets/amct_pdf_files` directory.
- Navigate to the `pdf_to_text` directory.
- Run the script: `python pdf2text.py`
- The converted TXT files will be saved in the `datasets/amct_text_files` directory.
