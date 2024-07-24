# BioHackton


## Description

This tool predicts if a given protein sequence is positive or negative to NES and optionally shows the contribution of each amino acid to the prediction.

## How to Use

1. **Clone the Repository or Download the Code**

   git clone BioHackton
   cd BioHackton
   \`\`\`

2. **Install Required Packages**

   pip install -r requirements.txt

3. **Run the Predictor**

   To check if a sequence is positive or negative, run the following command:

   python nes.py <sequence>

   Example:

   python nes.py MKTIIALSYIFCLVFAD

   If you also want to see how much each amino acid contributed to the decision of positive or negative by calculate the Saliency map of each amino acid , add `plot` at the end:

   python nes.py <sequence> plot

   Example:

   python nes.py MKTIIALSYIFCLVFAD plot

## Notes

- \`<sequence>\` should be your protein sequence in uppercase letters.
- The \`plot\` option will generate a saliency map showing the contribution of each amino acid to the prediction.

