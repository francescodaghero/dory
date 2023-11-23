# C_Parser.py
# Alessio Burrello <alessio.burrello@unibo.it>
# 
# Copyright (C) 2019-2020 University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Libraries
import json
import os
import numpy as np
from collections import OrderedDict
import shutil

# DORY modules
from dory.Parsers.Parser_HW_to_C import Parser_HW_to_C
from dory.Hardware_targets.PULP.GAP9_TVM import Layer2D_template_writer as Layer2D_writer
from dory.Hardware_targets.PULP.GAP9_TVM.TemplateWriter import TemplateWriter
import dory.Hardware_targets.PULP.Backend_Kernels.BackendKernelsAdapter as BackendKernelsAdapter
import dory.Utils.Templates_writer.writer_utils as utils


class C_Parser(Parser_HW_to_C):

    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, config_file, config_file_dir, verbose_level, perf_layer, precision_library, app_directory, n_inputs=1):

        file_path = self.get_file_path()
        with open(os.path.join(file_path, "HW_description.json")) as f:
            HW_description = json.load(f)
        self.precision_library = C_Parser._auto_precision_library(graph) if precision_library == "auto" else precision_library
        self.source_Constant_bits_library = config_file["BNRelu_bits"]
        self.config_file = config_file
        super().__init__(graph, os.path.join(config_file_dir, os.path.dirname(config_file["onnx_file"])), HW_description, verbose_level, perf_layer, "Makefile", app_directory, n_inputs)
        try:
            db = HW_description['double_buffering']
        except KeyError:
            print("C_Parser_PULP: Key 'double_buffering' not found in HW_description.json - setting to 2")

            db = 2
        self.double_buffering = db

    def get_file_path(self):
        return "/".join(os.path.realpath(__file__).split("/")[:-1])

    @staticmethod
    def _auto_precision_library(graph):
        precision_library = "8bit"
        for node in graph:
            if "Addition" not in node.name and "Pool" not in node.name:
                if node.get_parameter('output_activation_bits') < 8 or node.get_parameter('input_activation_bits') < 8 or node.get_parameter('weight_bits') < 8:
                    precision_library = 'mixed-sw'
            else:
                if node.get_parameter('output_activation_bits') < 8 or node.get_parameter('input_activation_bits') < 8:
                    precision_library = 'mixed-sw'
        return precision_library

    def node_backend_library(self, node):
        return self.precision_library

    def copy_backend_files(self, node, backend_library):
        if backend_library == "8bit":
            backendKernelsAdapter = BackendKernelsAdapter.PulpNNAdapter("pulp-nn", node, self.source_Constant_bits_library)
        elif backend_library == "mixed-sw":
            backendKernelsAdapter = BackendKernelsAdapter.PulpMixedAdapter("pulp-nn-mixed", node, self.source_Constant_bits_library, "sw")
        elif backend_library == "mixed-hw":
            backendKernelsAdapter = BackendKernelsAdapter.PulpMixedAdapter("pulp-nn-mixed", node, self.source_Constant_bits_library, "hw")
        else:
            raise ValueError(f"Unrecognised backend library: {backend_library}")

        for file in backendKernelsAdapter.get_src_files():
            shutil.copy(file, self.src_dir)

        for file in backendKernelsAdapter.get_inc_files():
            shutil.copy(file, self.inc_dir)

    def l2_template_keywords(self,tk, node, backend_library):
        return Layer2D_writer.print_template_layer(tk, node, backend_library, double_buffering=self.double_buffering)

    def l2_template_mapping(self, node, backend_library):
        """ TVM expects only the source file"""
        tmpl_c = self.l2_c_template(node, backend_library)
        return {
            os.path.join(self.src_dir, node.prefixed_name + ".c"): os.path.join(self.tmpl_dir, tmpl_c),
            #os.path.join(self.inc_dir, node.prefixed_name + ".h"): os.path.join(self.tmpl_dir, "layer_L2_h_template.h"),
        }
    

    def l2_c_template(self, node, backend_library):
        if "Pool" in node.op_type:
            if(backend_library == '1D_Conv'):
                return "pooling_layer_1D_template.c"
            else:
                return "layer_L2_c_pooling_template.c"
        elif "Add" in node.op_type:
            if(backend_library == '1D_Conv'):
                return "add_layer_1D_template.c"
            else:
                return "layer_L2_c_addition_template.c"
        else:
            return "layer_L2_c_conv_template.c"

    def mapping_layers_to_C_files(self) -> list:
        """Similar to the PULP one, but without writing to files"""
        print("\nMapping the layers files to their templates and copying the kernels associated.")
        n_memory_levels = self.HW_description['memory']['levels']
        assert len(self.HWgraph) == 1, "Expected only one node"

        #for i, node in enumerate(self.HWgraph):
        node = self.HWgraph[0]
        backend_library = self.node_backend_library(node)
        c_files = list()
        #self.copy_backend_files(node, backend_library)

        if n_memory_levels > 2 and (node.L3_input != 0 or (node.tiling_dimensions["L3"]["output_dimensions"] != node.tiling_dimensions["L2"]["output_dimensions"]) or (node.tiling_dimensions["L3"]["weights_dimensions"] != node.tiling_dimensions["L2"]["weights_dimensions"])):
            raise NotImplementedError("Currently not supported")
            tk = Layer2D_writer.print_template_layer_L3(node)
            TemplateWriter.write(tk, {os.path.join(self.src_dir, node.prefixed_name + ".c"): os.path.join(self.tmpl_dir, "layer_L3_c_template.c"),
                                      os.path.join(self.inc_dir, node.prefixed_name + ".h"): os.path.join(self.tmpl_dir, "layer_L3_h_template.h")})
            if node.tiling_dimensions["L3"]["input_dimensions"][1] > node.tiling_dimensions["L2"]["input_dimensions"][1]:
                node.tiling_dimensions["L2"]["output_dimensions"][1]  = int(np.floor((node.tiling_dimensions["L2"]["input_dimensions"][1] - node.kernel_shape[0] + node.strides[0]) / node.strides[0]))
            if node.tiling_dimensions["L3"]["output_dimensions"][1] > node.tiling_dimensions["L2"]["output_dimensions"][1]:
                node.tiling_dimensions["L2"]["input_dimensions"][1]   = node.tiling_dimensions["L2"]["output_dimensions"][1] * node.strides[0] + node.kernel_shape[0] - node.strides[0]
            node.name = node.name + "_L2"
            padding = node.pads
            node.pads = [0, padding[1], 0, padding[3]]
            tk = self.l2_template_keywords(node, backend_library)
            c_files += TemplateWriter.write(tk, self.l2_template_mapping(node, backend_library))
            node.name = node.name[:-3]
            if padding[0] > 0:
                node.name = node.name + "_L2_p_t"
                node.pads = [padding[0], padding[1], 0, padding[3]]
                tk = self.l2_template_keywords(node, backend_library)
                c_files += TemplateWriter.write(tk, self.l2_template_mapping(node, backend_library))
                node.name = node.name[:-1] + "b"
                node.pads = [0, padding[1], padding[2], padding[3]]
                node.tiling_dimensions["L2"]["input_dimensions"][1] -= (padding[2] - ((node.tiling_dimensions["L3"]["input_dimensions"][1] + padding[0] + padding[2]) - (node.tiling_dimensions["L3"]["output_dimensions"][1]* node.strides[0] + node.kernel_shape[0] - node.strides[0])))
                if node.tiling_dimensions["L1"]["input_dimensions"][1] > node.tiling_dimensions["L2"]["input_dimensions"][1]:
                    node.tiling_dimensions["L1"]["input_dimensions"][1] = node.tiling_dimensions["L2"]["input_dimensions"][1]
                if node.tiling_dimensions["L1"]["output_dimensions"][1] > node.tiling_dimensions["L2"]["output_dimensions"][1]:
                    node.tiling_dimensions["L1"]["output_dimensions"][1] = node.tiling_dimensions["L2"]["output_dimensions"][1]
                tk = self.l2_template_keywords(node, backend_library)
                c_files += TemplateWriter.write(tk, self.l2_template_mapping(node, backend_library))
                node.name = node.name[:-7]
        else:
            if node.tiling_dimensions["L2"]["input_dimensions"][2] == node.tiling_dimensions["L1"]["input_dimensions"][2]:
                node.tiling_dimensions["L1"]["output_dimensions"][2] = int((node.tiling_dimensions["L1"]["input_dimensions"][2] + (node.pads[1] + node.pads[3]) - node.kernel_shape[1] + node.strides[1]) / node.strides[1])
            if node.tiling_dimensions["L2"]["input_dimensions"][1] == node.tiling_dimensions["L1"]["input_dimensions"][1]:
                node.tiling_dimensions["L1"]["output_dimensions"][1] = int((node.tiling_dimensions["L1"]["input_dimensions"][1] + (node.pads[0] + node.pads[2]) - node.kernel_shape[0] + node.strides[0]) / node.strides[0])
            tk = self.create_hex_weights_files(node)
            tk = self.l2_template_keywords(tk, node, backend_library)
            c_files += TemplateWriter.write(tk, self.l2_template_mapping(node, backend_library))
        return c_files

    def create_hex_weights_files(self, node):
        #TODO : Capire  se l'accel. vuole un layout diverso. Altrimenti i pesi non cambiano
        print(f"\nGenerating weight string for {node.name}.")

        weights = np.array([])
        for val in node.constant_names:
            if val in ["weights", "bias","k", "lambda"]:
                weights = np.concatenate((weights,node.__dict__[val]["value"]))
        tk = OrderedDict([])
        tk['weights_vectors'] = utils.print_test_vector(weights, 'char')
        tk['weights_dimensions'] = weights.shape[0]
        return tk