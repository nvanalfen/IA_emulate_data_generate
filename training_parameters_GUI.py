import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, \
                            QPushButton, QCheckBox, QFrame, QRadioButton
import numpy as np

class ParameterGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setGeometry(100, 100, 400, 400)
        self.setWindowTitle('Simple GUI')

        main_layout = QVBoxLayout()

        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel('Decimal Tolerance:'))
        self.tol_input = QLineEdit("3")
        tol_row.addWidget(self.tol_input)
        main_layout.addLayout(tol_row)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        main_layout.addWidget(line)

        # First set of number inputs
        self.start_input = QLineEdit("0")
        self.stop_input = QLineEdit("1")
        self.step_input = QLineEdit("0.1")

        pre_first_row_layout = QHBoxLayout()
        pre_first_row_layout.addWidget(QLabel('Central Alignment Strengths:'))
        main_layout.addLayout(pre_first_row_layout)

        first_row_layout = QHBoxLayout()
        first_row_layout.addWidget(QLabel('μ:'))
        first_row_layout.addWidget(QLabel('Start:'))
        first_row_layout.addWidget(self.start_input)
        first_row_layout.addWidget(QLabel('Stop:'))
        first_row_layout.addWidget(self.stop_input)
        first_row_layout.addWidget(QLabel('Step:'))
        first_row_layout.addWidget(self.step_input)

        main_layout.addLayout(first_row_layout)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        main_layout.addWidget(line)

        pre_second_row_layout = QHBoxLayout()
        pre_second_row_layout.addWidget(QLabel('Satellite Alignment Strengths:'))
        main_layout.addLayout(pre_second_row_layout)

        # Checkbox for constant
        self.constant_checkbox = QCheckBox('Constant')
        self.constant_checkbox.stateChanged.connect(self.on_constant_checkbox_changed)
        main_layout.addWidget(self.constant_checkbox)

        # Second set of number inputs (depends on checkbox state)
        self.start_input2 = QLineEdit("0")
        self.stop_input2 = QLineEdit("1")
        self.step_input2 = QLineEdit("0.1")

        self.second_row_layout = QHBoxLayout()
        self.sat_label = QLabel('a:')
        self.second_row_layout.addWidget(self.sat_label)
        self.second_row_layout.addWidget(QLabel('Start:'))
        self.second_row_layout.addWidget(self.start_input2)
        self.second_row_layout.addWidget(QLabel('Stop:'))
        self.second_row_layout.addWidget(self.stop_input2)
        self.second_row_layout.addWidget(QLabel('Step:'))
        self.second_row_layout.addWidget(self.step_input2)

        main_layout.addLayout(self.second_row_layout)

        # Third set of number inputs (depends on checkbox state)
        self.start_input3 = QLineEdit()
        self.stop_input3 = QLineEdit()
        self.step_input3 = QLineEdit()

        self.third_row_layout = QHBoxLayout()
        self.third_row_layout.addWidget(QLabel('gamma:'))
        self.third_row_layout.addWidget(QLabel('Start:'))
        self.third_row_layout.addWidget(self.start_input3)
        self.third_row_layout.addWidget(QLabel('Stop:'))
        self.third_row_layout.addWidget(self.stop_input3)
        self.third_row_layout.addWidget(QLabel('Step:'))
        self.third_row_layout.addWidget(self.step_input3)

        main_layout.addLayout(self.third_row_layout)

        # Lower and upper bounds for the third set of inputs
        self.lower_bound_input = QLineEdit()
        self.upper_bound_input = QLineEdit()
        self.r_ratio = QLineEdit("0.9")

        self.bounds_row_layout = QHBoxLayout()
        self.bounds_row_layout.addWidget(QLabel('Lower Bound:'))
        self.bounds_row_layout.addWidget(self.lower_bound_input)
        self.bounds_row_layout.addWidget(QLabel('Upper Bound:'))
        self.bounds_row_layout.addWidget(self.upper_bound_input)
        self.bounds_row_layout.addWidget(QLabel('r/r_vir:'))
        self.bounds_row_layout.addWidget(self.r_ratio)
        self.bounds_row_layout.setEnabled(False)  # Initially disabled

        main_layout.addLayout(self.bounds_row_layout)

        # Number of loops input on the same line
        self.num_loops_input = QLineEdit("100")
        num_loops_layout = QHBoxLayout()
        num_loops_layout.addWidget(QLabel('Number of Inner Loops:'))
        num_loops_layout.addWidget(self.num_loops_input)
        main_layout.addLayout(num_loops_layout)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        main_layout.addWidget(line)

        # Other administrative variables
        pre_administrative_row_layout = QHBoxLayout()
        pre_administrative_row_layout.addWidget(QLabel('Catalog:'))
        main_layout.addLayout(pre_administrative_row_layout)

        self.bolplanck_radio = QRadioButton('Bolshoi-Planck')
        self.multidark_radio = QRadioButton('MultiDark')
        self.radio_layout = QHBoxLayout()
        self.radio_layout.addWidget(self.bolplanck_radio)
        self.radio_layout.addWidget(self.multidark_radio)
        main_layout.addLayout(self.radio_layout)
        self.bolplanck_radio.setChecked(True)

        self.logspace_start_input = QLineEdit("-1")
        self.logspace_stop_input = QLineEdit("1.2")
        self.logspace_edge_input = QLineEdit("15")

        logbin_layout = QHBoxLayout()
        logbin_layout.addWidget(QLabel('Log Bins:'))
        logbin_layout.addWidget(QLabel('Start:'))
        logbin_layout.addWidget(self.logspace_start_input)
        logbin_layout.addWidget(QLabel('Stop:'))
        logbin_layout.addWidget(self.logspace_stop_input)
        logbin_layout.addWidget(QLabel('Edges:'))
        logbin_layout.addWidget(self.logspace_edge_input)
        main_layout.addLayout(logbin_layout)

        self.data_directory_input = QLineEdit("data")

        pre_administrative_row_layout = QHBoxLayout()
        pre_administrative_row_layout.addWidget(QLabel('Data Output Directory:'))
        pre_administrative_row_layout.addWidget(self.data_directory_input)
        main_layout.addLayout(pre_administrative_row_layout)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        main_layout.addWidget(line)

        # Output label, text input, and ".npy" label
        main_layout.addWidget(QLabel('Output Filename'))
        output_layout = QHBoxLayout()
        self.output_text_input = QLineEdit()
        self.npy_label = QLabel('.npy')
        output_layout.addWidget(self.output_text_input)
        output_layout.addWidget(self.npy_label)
        main_layout.addLayout(output_layout)

        # Generate button
        generate_button = QPushButton('Generate')
        generate_button.clicked.connect(self.generate_param_file)
        main_layout.addWidget(generate_button)

        self.setLayout(main_layout)

        self.constant_checkbox.setChecked(True)

    def on_constant_checkbox_changed(self, state):
        if state == 2:  # Checked
            self.second_row_layout.setEnabled(True)
            self.third_row_layout.setEnabled(False)
            self.bounds_row_layout.setEnabled(False)
            self.start_input3.setEnabled(False)
            self.stop_input3.setEnabled(False)
            self.step_input3.setEnabled(False)
            self.lower_bound_input.setEnabled(False)
            self.upper_bound_input.setEnabled(False)
            self.sat_label.setText('μ:')
            self.r_ratio.setEnabled(False)
        else:
            self.second_row_layout.setEnabled(False)
            self.third_row_layout.setEnabled(True)
            self.bounds_row_layout.setEnabled(True)
            self.start_input3.setEnabled(True)
            self.stop_input3.setEnabled(True)
            self.step_input3.setEnabled(True)
            self.lower_bound_input.setEnabled(True)
            self.upper_bound_input.setEnabled(True)
            self.sat_label.setText('a:')
            self.r_ratio.setEnabled(True)

    def in_r_bound(self, a, gamma):
        lowest = self.lower_bound_input.text()
        if lowest == '' or not self.lower_bound_input.isEnabled():
            lowest = -np.inf
        else:
            lowest = float(lowest)
        
        highest = self.upper_bound_input.text()
        if highest == '' or not self.upper_bound_input.isEnabled():
            highest = np.inf
        else:
            highest = float(highest)
        r_ratio = float( self.r_ratio.text() )
        val =  a * ( r_ratio**gamma )

        #print(f"val: {val}, a: {a}, gamma: {gamma}, lowest: {lowest}, highest: {highest}")

        return val >= lowest and val <= highest

    def build_param_dict(self):
        params = {}

        tol = int(self.tol_input.text())

        # Add cen mu values
        start = float(self.start_input.text())
        stop = float(self.stop_input.text()) + float(self.step_input.text())
        step = float(self.step_input.text())
        cen_mus = np.array( [ round(el, tol) for el in np.arange(start, stop, step) ] )
        params['cen_mus'] = cen_mus

        # Add sat params
        start = float(self.start_input2.text())
        stop = float(self.stop_input2.text()) + float(self.step_input2.text())
        step = float(self.step_input2.text())
        sat_a = np.array( [ round(el, tol) for el in np.arange(start, stop, step) ] )

        sat_params = []

        if not self.constant_checkbox.isChecked():
            start = float(self.start_input3.text())
            stop = float(self.stop_input3.text()) + float(self.step_input3.text())
            step = float(self.step_input3.text())
            sat_gamma = np.array( [ round(el, tol) for el in np.arange(start, stop, step) ] )

            for a in sat_a:
                for gamma in sat_gamma:
                    if self.in_r_bound(a, gamma):
                        sat_params.append( (a, gamma) )
            sat_params = np.array(sat_params)
        else:
            sat_params = sat_a

        params['sat_params'] = sat_params

        # Add Constant status
        params['constant'] = self.constant_checkbox.isChecked()

        # Add inner loops
        params['inner_runs'] = int(self.num_loops_input.text())

        # Add catalog
        if self.bolplanck_radio.isChecked():
            params['catalog'] = 'bolplanck'
        else:
            params['catalog'] = 'multidark'

        # Add logspace bins
        logspace_start = float(self.logspace_start_input.text())
        logspace_stop = float(self.logspace_stop_input.text())
        logspace_edge = int(self.logspace_edge_input.text())
        rbins = np.logspace(logspace_start, logspace_stop, logspace_edge)
        params['rbins'] = rbins

        # Add data directory
        params['output_dir'] = self.data_directory_input.text()

        return params

    def generate_param_file(self):
        output_filename = self.output_text_input.text() + '.npy'
        if output_filename == '.npy':
            print("Please enter an output filename.")
            return
        
        try:
            print("Generating output to:", output_filename)
            # Write param dict
            params = self.build_param_dict()
            np.save(output_filename, params)
        except:
            print("Error generating output file.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ParameterGUI()
    ex.show()
    sys.exit(app.exec_())
