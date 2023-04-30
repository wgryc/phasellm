# ResearchLLM

An autonomous statistics helper that converts your natural language queries about a data set to insights.

- Converts natural language questions to Python code
- Runs code locally without sharing data with third parties (just shares metadata)
- Interpets results
- Provide access to underlying Python code for audit and review

[2-minute demo below:](https://www.youtube.com/watch?v=-fzFCii6UoA)
[![ResearchLLM screenshot](screenshot.png)](https://www.youtube.com/watch?v=-fzFCii6UoA)

Please note that we originally launched this as *ResearchLLM* and have since renamed the demo to *ResearchLLM*. Apologies for any confusion!

## 🚨🚨 WARNING: Runs LLM-Generated Python Code

This product will run LLM-generated Python code on your computer/server. We highly recommend sandboxing the code or running this on a server that doesn't contain any sensitive information or processes.

## Installation and Setup

### Installation

Clone the GitHub repository and navigate to the folder containing this README.md file. Install the relevant packages (including PhaseLLM):

```
pip install -r requirements.txt
```

Next, make sure you edit the `researchllm.py` file to include the proper API keys. You'll find these around line 19:
```python
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = ClaudeWrapper(ANTHROPIC_API_KEY)
```

You can change the model type from ClaudeWrapper to other PhaseLLM wrappers. Make sure to update your API key accordingly, either via an environment variable or directly in the code.

### Running With Sample Data

Start a Python REPL (i.e, run `python` in the folder with all the files from this repo) and then type the following:

```
from frontend import *
run() # Or, run('0.0.0.0', 80) for a public server
```

Running `run()` will launch the server on 127.0.0.1:5000 (i.e., the default Flask setting).

### Running With Your Own Custom Data

Running this with your own data only requires a few simple changes to `frontend.py`. Around Line 20, you'll see the following comments:
```python
##########################################################################
#
# DATA SET SETUP (START)
# Please review the code below to set up your own data set for analysis.
#
```

All the instructions are there, but we repeat them here for your convenience. You will have to update the two variables below:
```python
DATA_SETUP_INTRO = "I am researching the relationship between income and sociodemographic census info."
DATA_FILE_LOC = "incomes.csv"
```

`DATA_SETUP_INTRO` should be one short sentence on the context of your data, while `DATA_FILE_LOC` is the location of the file you're loading.

If you are *not* using a CSV file, you can also load the DataFrame via a few lines down:
```python
df = pd.read_csv(DATA_FILE_LOC)
```

Replace the line above with your custom loader (e.g., read_excel() or something else). The `df` variable needs to be a Pandas dataframe for this to work.

## Sample Data Files and Credits

The sample data set included in this project and in the demo video is from the 1994 US census. It was put together by Ron Kohavi and is [available on Kaggle](https://www.kaggle.com/datasets/uciml/adult-census-income?select=adult.csv).

The other data set referenced in our code is [also on Kaggle](https://www.kaggle.com/datasets/new-york-city/nypd-motor-vehicle-collisions), focusing on motor vehicle collisions in New York City. We didn't include it in the repository as it's about 500MB in size. It's a good alternative to the census data above because it contains location data (latitude, longitude pairs), leading to some really interesting analysis options.
