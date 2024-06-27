import requests
from PIL import Image
import streamlit as st
import os
from pix2tex.cli import LatexOCR
from sympy.parsing.latex import parse_latex
from sympy.parsing.latex.errors import LaTeXParsingError
from sympy.solvers import solve
from sympy import symbols
import tqdm
import io
import requests

if 'latex_code' not in st.session_state:
    st.session_state['latex_code'] = ''

if 'variables' not in st.session_state:
    st.session_state['variables'] = {}

if 'equation' not in st.session_state:
    st.session_state['equation'] = None

if 'solution' not in st.session_state:
    st.session_state['solution'] = None

if 'lhs' not in st.session_state:
    st.session_state['lhs'] = None

url = 'https://github.com/lukas-blecher/LaTeX-OCR/releases/latest'

def update_latex(updated_code):
    st.session_state['latex_code'] = updated_code
    print("🚀 ~ st.session_state['latex_code']:", st.session_state['latex_code'])

def extract_variables():
    
    try:
      # Extract the prediction from the textbox
      prediction = st.session_state['latex_code']
      print("🚀 ~ LaTeX prediction:", prediction)
      
      # Preprocess the LaTeX string to remove or replace \biggr, \biggl, \bigl, \bigr
      prediction_processed = prediction
      prediction_processed = prediction_processed.replace(r'{\Big(}', r'\left(')
      prediction_processed = prediction_processed.replace(r'{\big(}', r'\left(')
      prediction_processed = prediction_processed.replace(r'{\Bigg(}', r'\left(')
      prediction_processed = prediction_processed.replace(r'{\bigg(}', r'\left(')
      prediction_processed = prediction_processed.replace(r'{\Bigg[}', r'\left[')
      prediction_processed = prediction_processed.replace(r'{\bigg[}', r'\left[')
      prediction_processed = prediction_processed.replace(r'{\Big[}', r'\left[')
      prediction_processed = prediction_processed.replace(r'{\big[}', r'\left[')

      prediction_processed = prediction_processed.replace(r'{\Big)}', r'\right)')
      prediction_processed = prediction_processed.replace(r'{\big)}', r'\right)')
      prediction_processed = prediction_processed.replace(r'{\Bigg)}', r'\right)')
      prediction_processed = prediction_processed.replace(r'{\bigg)}', r'\right)')
      prediction_processed = prediction_processed.replace(r'{\Bigg]}', r'\right]')
      prediction_processed = prediction_processed.replace(r'{\bigg]}', r'\right]')
      prediction_processed = prediction_processed.replace(r'{\Big]}', r'\right]')
      prediction_processed = prediction_processed.replace(r'{\big]}', r'\right]')
      print("🚀 ~ Processed LaTeX prediction:", prediction_processed)

      # Parse the LaTeX equation
      equation = parse_latex(rf"{prediction_processed}")
      print("🚀 ~ SymPy equation:", equation)
      st.session_state['equation'] = equation
      
      # Extract and display the free symbols
      variables_set = equation.free_symbols
      print("🚀 ~ Variables extracted:", variables_set)

      unwanted_symbols = {'bigr', 'biggl', 'biggr', 'bigl', 'big', 'Big', 'Bigr', 'Biggl', 'Biggr', 'Bigl', 'bigg', 'Bigg'}
      
      filtered_variables_set = {var for var in variables_set if var.name not in unwanted_symbols}

      # Extract the left-hand side and remove it from the set of variables
      lhs = equation.lhs
      st.session_state['lhs'] = lhs
      lhs_set = {lhs}
      filtered_variables_set -= lhs_set
      filtered_variables_list = list(filtered_variables_set)
      print("🚀 ~ Variables extracted:", filtered_variables_list)

      # Save variables to session state
      st.session_state['variables'] = {str(var): 0.0 for var in filtered_variables_list}

      # Show input fields for the filtered variables
      # showVariableInputs(filtered_variables_list)
        
    except LaTeXParsingError as e:
        st.error("LaTeX parsing error", e)
    except Exception as e:
        st.error("Error", e)
  
def calculate():
    try:
        # Retrieve the values for substitution from session state
        substitution_dict = {symbols(var): st.session_state['variables'][var] for var in st.session_state['variables']}
        print("🚀 ~ Substitution dictionary:", substitution_dict)

        if st.session_state['equation'] is None:
            st.error("Equation not found. Please extract variables first.")
            return
        print("🚀 ~ st.session_state['equation']:", st.session_state['equation'])
        substituted_eq = st.session_state['equation'].subs(substitution_dict)
        print("🚀 ~ Substituted equation:", substituted_eq)
        
        # Extract the left-hand side (variable t) and solve for it
        lhs = st.session_state['equation'].lhs
        solutions = solve(substituted_eq, lhs)
        st.session_state['solution'] = solutions[0]

    except Exception as e:
        print("Error in calculation:", e)
        st.error("Error in calculation")

def get_latest_tag():
    r = requests.get(url)
    tag = r.url.split('/')[-1]
    if tag == 'releases':
        return 'v0.0.1'
    return tag


def download_as_bytes_with_progress(url: str, name: str = None) -> bytes:
    # source: https://stackoverflow.com/questions/71459213/requests-tqdm-to-a-variable
    resp = requests.get(url, stream=True, allow_redirects=True)
    total = int(resp.headers.get('content-length', 0))
    bio = io.BytesIO()
    if name is None:
        name = url
    with tqdm.tqdm(
        desc=name,
        total=total,
        unit='b',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            bar.update(len(chunk))
            bio.write(chunk)
    return bio.getvalue()


def download_checkpoints_custom_path():
    tag = get_latest_tag()
    path = './weights'
    print('download weights', tag, 'to path', path)
    weights = 'https://github.com/lukas-blecher/LaTeX-OCR/releases/download/%s/weights.pth' % tag
    resizer = 'https://github.com/lukas-blecher/LaTeX-OCR/releases/download/%s/image_resizer.pth' % tag
    for url, name in zip([weights, resizer], ['weights.pth', 'image_resizer.pth']):
        file = download_as_bytes_with_progress(url, name)
        open(os.path.join(path, name), "wb").write(file)
        
class CustomLatexOCR(LatexOCR):
    def __init__(self):
        download_checkpoints_custom_path()
        super().__init__()

def main():
    st.set_page_config(page_title='LaTeX-OCR')
    st.title('LaTeX OCR')
    st.markdown('Convert images of equations to corresponding LaTeX code and perform calculations.\n\nThis is based on the `pix2tex` module. Check it out [![github](https://img.shields.io/badge/LaTeX--OCR-visit-a?style=social&logo=github)](https://github.com/lukas-blecher/LaTeX-OCR)')
    model = CustomLatexOCR()

    uploaded_file = st.file_uploader('Upload an image of an equation', type=['png', 'jpg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image)

        if st.button('Convert'):
            with st.spinner('Computing...'):
                latex_code = model(image)
                if latex_code:
                    st.session_state['latex_code'] = latex_code
                else:
                    st.error('Error converting the image to LaTeX.')

    if st.session_state['latex_code']:
        st.markdown(f'$$ {st.session_state["latex_code"]} $$')
        st.code(st.session_state['latex_code'], language='latex')

        with st.expander('Edit LaTeX Code'):
          col1, col2 = st.columns([0.85, 0.15], vertical_alignment="bottom")
          with col1:
              editable_latex_code = st.text_area('Edit:', value=st.session_state['latex_code'], key='editable_latex_code')
          with col2:
              st.button('Update', on_click=update_latex, args=[editable_latex_code])
        col1, col2 = st.columns([0.15, 0.85], vertical_alignment="bottom")
        with col1:
              st.button('Confirm', on_click=extract_variables)
        with col2:
              st.button('Clear', on_click=lambda: st.session_state.__delitem__('latex_code'))

    if st.session_state['variables']:
        st.markdown('##### \n\n ### Input Variables')
        for var in st.session_state['variables']:
            st.session_state['variables'][var] = st.number_input(f'{var}', value=st.session_state['variables'][var], step=1.0)
        st.button('Calculate', on_click=calculate)

    if st.session_state['solution']:
        st.markdown(f'###### \n\n ##### Solution for {st.session_state["lhs"]}:')
        st.code(f'{st.session_state["solution"]}', language='python')

if __name__ == '__main__':
    main()
