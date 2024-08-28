import os
from colorama import Fore
import sys
from fpdf import FPDF

# Add the directory containing `predictions.py` to the system path
sys.path.append("C:/(D)/Prajith K/Studies/Projects/Final Project/Project/Multi_Model_Medical_Image_Classification_Detection/Training and Testing codes")

# Now you can import the predict function
from predictions import predict

def load_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    dataset = []
    for body in os.listdir(path):
        body_part = body
        path_p = os.path.join(path, body_part)
        for lab in os.listdir(path_p):
            label = lab
            path_l = os.path.join(path_p, label)
            for img in os.listdir(path_l):
                img_path = os.path.join(path_l, img)
                # Check if the path is a file before appending
                if os.path.isfile(img_path):
                    dataset.append(
                        {
                            'body_part': body_part,
                            'label': label,
                            'image_path': img_path,
                            'image_name': img
                        }
                    )
    return dataset

def reportPredict(dataset):
    total_count = 0
    status_count = 0

    print(Fore.YELLOW +
          '{0: <28}'.format('Name') +
          '{0: <14}'.format('Part') +
          '{0: <20}'.format('Status') +
          '{0: <20}'.format('Predicted Status'))
    for img in dataset:
        fracture_predict = predict(img['image_path'], img['body_part'])
        if img['label'] == fracture_predict:
            status_count += 1
            color = Fore.GREEN
        else:
            color = Fore.RED
        print(color +
              '{0: <28}'.format(img['image_name']) +
              '{0: <14}'.format(img['body_part']) +
              '{0: <20}'.format(img['label']) +
              '{0: <20}'.format(fracture_predict))

    status_accuracy = (status_count / len(dataset)) * 100
    print(Fore.BLUE + 'Status accuracy: ' + str("%.2f" % status_accuracy) + '%')
    return status_accuracy

# Define a function to create a PDF report
def create_pdf_report(dataset, filename, status_accuracy):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Predictions Report", ln=True, align="C")
    pdf.cell(200, 10, ln=True)  # Add a line break

    # Add table headers
    pdf.cell(70, 10, "Name", border=1)
    pdf.cell(40, 10, "Part", border=1)
    pdf.cell(40, 10, "Status", border=1)
    pdf.cell(40, 10, "Predicted Status", border=1)
    pdf.ln()

    # Add data rows
    for img in dataset:
        pdf.cell(70, 10, img['image_name'], border=1)
        pdf.cell(40, 10, img['body_part'], border=1)
        pdf.cell(40, 10, img['label'], border=1)
        pdf.cell(40, 10, predict(img['image_path'], img['body_part']), border=1)
        pdf.ln()

    # Add status accuracy
    pdf.cell(200, 10, ln=True)  # Add a line break
    pdf.cell(200, 10, txt="Status Accuracy: " + str("%.2f" % status_accuracy) + "%", ln=True, align="C")

    # Save the PDF
    pdf.output(filename)

# Update the path to the test directory
test_dir = "C:/(D)/Prajith K/Studies/Projects/Final Project/backups/Bone-Fracture-Detection-master/test"
if os.path.exists(test_dir):
    dataset = load_path(test_dir)
    status_accuracy = reportPredict(dataset)
    
    # Create and save the PDF report
    pdf_filename = "predictions_report.pdf"
    create_pdf_report(dataset, pdf_filename, status_accuracy)
    
    print(Fore.GREEN + f"PDF report generated: {pdf_filename}")
else:
    print(Fore.RED + f"The test directory does not exist: {test_dir}")
