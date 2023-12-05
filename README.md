# financial_nlp_nlu
Given the quick pace of decision-making by investors, creditors, and lenders in the shifting landscape of financial markets, it is crucial to stay informed. Among the several tactics offered by analysts, one key practice for comprehending the complexities of the financial industry stands out: studying the detail included in financial reports produced by big financial organizations such as J.P.Morgan and Goldman Sachs.

The primary aim of this project is to facilitate financial reports analyses, with a focus on expediting information extraction and enabling pretty fast and clear visualization. A goal-oriented and fast information extraction process from important financial reports can be a game changer and enable much wiser and more strategic decisions in the financial domain.

# Key features
- [Database](#database)
- 
## Database
In order to be able to compare our gained insight with what really happened in the financial market and also be working with recent data we opted for 17 financial reports published in the 1st semester of 2023 which are: 
- Goldman Sachs_Caution Heavy Fog
- Unicredit_Economics chartbook Q2
- bnp_parisbas global view 2023
- UBS_alternative-investments-improving-portfolio-performance
- Goldman Sachs_global view 2023
- kkr_global view 2023
- Moodys_Weekly-Market-Outlook
- CACIB_-Monde-Hebdo
- Erste_Week Ahead
- ScotiaBank_global week ahead
- Commerz_European Sunrise
- ING_FX Talking July22
- 12 31 2022_jpmorgan_asset management Q1 2023
- 16 11 2022_Goldman Sachs_global outlook
- 2022_jpmorgan_private banking global view 2023
- 29 07 2022_Goldman Sachs_exemple analyse macro economique goldman sachs
- 29 07 2022_Goldman Sachs_recession
### Test Extraction
For this particular scenario, OCR proved to be time-consuming, and it struggled notably with multi-column text layouts and irregular text orientations, both of which are commonly found in financial reports. For that I chose to use PyMuPDF also known as Fitz which is a flexible python library offering advanced bunch of functionalities to deal with pdf documents including fast text and image extraction with consideration of internal document structure, access to metadata and perform text search.

### Text cleaning
Considering that inconsistent noisy data, such as variations in spelling, punctuation, formatting, irrelevant symbols and special characters can potentially confuse various models and lead to poor results, I consider the text cleaning step pivotal for the success of this project.
The preprocess_text function in our script will :
- 
### Dataframe Creation
