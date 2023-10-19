/*
 * layer_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Francesco Conti <f.conti@unibo.it>
 *
 * Copyright (C) 2018-2020 University of Bologna
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 */

#include "pmsis.h"
#include "dory_get_tile.h"
#include "dory_dma.h"
#include "pulp_nn_kernels.h"
#include "tile_index.h"
#include "layer.h"
#include "double_buffer.h"

% if ULTRA_VERBOSE:
#define VERBOSE_PRINT(...) printf(__VA_ARGS__)
% endif


static const TileIndex index_end = {
  .height = ${tile_dim_h},
  .width = ${tile_dim_w},
  .input_channel = ${tile_dim_nif},
  .output_channel = ${tile_dim_nof},
};

static const Layer body = {
  .input = {
    .height = ${x_tile_size_h},
    .width = ${x_tile_size_w},
    .channel = ${x_tile_size_nif},
    .channel_size = ${x_tile_size_nif_byte}
  },
  .output = {
    .height = ${y_tile_size_h},
    .width = ${y_tile_size_w},
    .channel = ${y_tile_size_nof},
    .channel_size = ${y_tile_size_nof_byte}
  },
  .weights = {
    .output_channel = ${W_tile_size_nof},
    .input_channel = ${W_tile_size_nif},
    .input_channel_size = ${W_tile_nif_byte}
  }
};

static const Layer border = {
  .input = {
    .height = ${x_tile_size_h_last},
    .width = ${x_tile_size_w_last},
    .channel = ${x_tile_size_nif_last},
    .channel_size = ${x_tile_size_nif_byte_last}
  },
  .output = {
    .height = ${y_tile_size_h_last},
    .width = ${y_tile_size_w_last},
    .channel = ${y_tile_size_nof_last},
    .channel_size = ${y_length_nof_byte_last}
  },
  .weights = {
    .output_channel = ${W_tile_size_nof_last},
    .input_channel = ${W_tile_size_nif_last},
    .input_channel_size = ${W_tile_size_nif_byte_last}
  }
};


static void load_input_async(Layer tile, Layer body, Layer layer, TileIndex index) {
  // additionally overlap by padding for the first tile after a border one
  // this because in the first tile we use less pixels from x_buffer, since we have the ones of padding
  const int x_offset_h = index.height > 0 ? layer.padding.top : 0;
  const int x_offset_w = index.width > 0 ? layer.padding.left : 0;

  dma_transfer_async((DmaTransferConf) {
    .ext = dory_get_tile_3d(layer.addr.input,
                            index.height, index.width, index.input_channel,
                            body.input.height, body.input.width, body.input.channel,
                            layer.input.width, layer.input.channel,
                            ${conv_overlap1}, ${conv_overlap2}, 0,
                            x_offset_h, x_offset_w, 0,
                            ${x_data_size_byte}),
    .loc = tile.addr.input,
    .number_of_2d_copies = tile.input.height,
    .number_of_1d_copies = tile.input.width,
    .length_1d_copy = tile.input.channel_size,
    .hwc_to_chw = ${flag_DW},
    .stride_2d = ${x_stride_w_byte},
    .stride_1d = ${x_stride_c_byte},
    .dir = 1
  });
}

static void store_output_async(Layer tile, Layer body, Layer layer, TileIndex index) {
  dma_transfer_async((DmaTransferConf) {
    .ext = dory_get_tile_3d(layer.addr.output,
                            index.height, index.width, index.output_channel,
                            body.output.height, body.output.width, body.output.channel,
                            layer.output.width, layer.output.channel,
                            0, 0, 0,
                            0, 0, 0,
                            ${y_data_size_byte}),
    .loc = tile.addr.output,
    .number_of_2d_copies = tile.output.height,
    .number_of_1d_copies = tile.output.width,
    .length_1d_copy = tile.output.channel_size,
    .hwc_to_chw = 0,
    .stride_2d = ${y_stride_w_byte},
    .stride_1d = ${y_stride_c_byte},
    .dir = 0
  }); 
}

static void load_weights_async(Layer tile, Layer body, Layer layer, TileIndex index) {
  dma_transfer_async((DmaTransferConf) {
    .ext = dory_get_tile_3d(layer.addr.weights,
                            % if flag_DW == 0:
                            index.output_channel, 0, index.input_channel,
                            % else:
                            index.output_channel, 0, 0,
                            % endif
                            body.weights.output_channel, ${fs1}*${fs2}, body.weights.input_channel,
                            ${fs1}*${fs2}, ${nif},
                            0, 0, 0,
                            0, 0, 0,
                            ${W_data_size_byte}),
    .loc = tile.addr.weights,
    % if flag_DW == 0:
    .number_of_2d_copies = tile.weights.output_channel,
    .number_of_1d_copies = ${fs1 * fs2},
    .length_1d_copy = tile.weights.input_channel_size,
    % else:
    .number_of_2d_copies = 1,
    .number_of_1d_copies = 1,
    .length_1d_copy = (int) tile.weights.output_channel * ${W_data_size_byte} * ${fs1}*${fs2} / 8,
    % endif
    .hwc_to_chw = 0,
    .stride_2d = ${W_stride_nof_byte},
    .stride_1d = ${W_stride_hw_byte},
    .dir = 1
  });
  % if FLAG_BATCHNORM == 1:

  dma_transfer_1d_async((DmaTransferConf) {
    .ext = layer.addr.scale + ${k_tile_size_byte_transfer}*index.output_channel,
    .loc = tile.addr.scale,
    .length_1d_copy = tile.weights.output_channel * ${int(act_dim_bit/8)},
    .dir = 1
  });

  dma_transfer_1d_async((DmaTransferConf) {
    .ext = layer.addr.bias + ${lambda_tile_size_byte_transfer}*index.output_channel,
    .loc = tile.addr.bias,
    .length_1d_copy = tile.weights.output_channel * ${int(act_dim_bit/8)},
    .dir = 1
  });
  % endif
}

static void load_bias_async(Layer tile, Layer layer) {
  dma_transfer_1d_async((DmaTransferConf) {
    .ext = layer.addr.bias,
    .loc = tile.addr.bias,
    .length_1d_copy = ${b_size_byte},
    .dir = 1
  });
}

static void kernel(Layer tile, void * im2col, void * pwt_buffer) {
  % if flag_DW == 0 and optional_type == '8bit' and (fs1*fs2>1 or stride>1):
    pulp_nn_conv_Ho_parallel(
  % elif flag_DW == 0 and optional_type == '8bit' and fs1*fs2==1  and 'FullyConnected' not in func_name:
    pulp_nn_pointwise_HoWo_parallel(
  % elif flag_DW == 0 and optional_type == '8bit' and y_data_size_byte == 32 and ('FullyConnected' in func_name):
    pulp_nn_linear_out_32( 
  % elif flag_DW == 0 and optional_type == '8bit' and ('FullyConnected' in func_name):
    pulp_nn_linear( 
  % elif flag_DW == 0 and 'mixed' in optional_type  and ('Conv' in func_name):
    ${"x" if 'hw' in optional_type else ""}pulp_nn_conv_${data_type_x[0]}${x_data_size_byte}_${data_type_y[0]}${y_data_size_byte}_${data_type_weights[0]}${W_data_size_byte}(
  % elif flag_DW == 0 and 'mixed' in optional_type  and ('Gemm' in func_name or 'MatMul' in func_name or 'FullyConnected' in func_name) and y_data_size_byte == 32:
    ${"x" if 'hw' in optional_type else ""}pulp_nn_linear_${data_type_x[0]}${x_data_size_byte}_${data_type_y[0]}${y_data_size_byte}_${data_type_weights[0]}${W_data_size_byte}(
  % elif flag_DW == 0 and 'mixed' in optional_type  and ('Gemm' in func_name or 'MatMul' in func_name or 'FullyConnected' in func_name):
    pulp_nn_linear_${data_type_x[0]}${x_data_size_byte}_${data_type_y[0]}${y_data_size_byte}_${data_type_weights[0]}${W_data_size_byte}(
  % elif flag_DW == 1 and optional_type == '8bit' and fs1 == 3 and fs2 == 3 and stride==1:
    pulp_nn_depthwise_generic(
  % elif flag_DW == 1 and optional_type == '8bit' and fs1*fs2 < 4:
    pulp_nn_depthwise_generic_less_4_weights(
  % elif flag_DW == 1 and optional_type == '8bit':
    pulp_nn_depthwise_generic(
  % elif flag_DW == 1 and 'mixed' in optional_type:
    ${"x" if 'hw' in optional_type else ""}pulp_nn_depthwise_${data_type_x[0]}${x_data_size_byte}_${data_type_y[0]}${y_data_size_byte}_${data_type_weights[0]}${W_data_size_byte}(
  % endif
  % if 'Gemm' in func_name or 'FullyConnected' in func_name:
      tile.addr.input,
      % if has_bias:
      tile.addr.bias,
      % else:
      NULL,
      % endif
      tile.addr.output,
      tile.addr.weights,
      % if FLAG_BATCHNORM == 1 and y_data_size_byte != 32:
      tile.addr.scale, tile.addr.bias,
      % elif y_data_size_byte != 32:
      0, 0,
      % endif
      % if y_data_size_byte != 32:
      ${out_mul}, ${out_shift},
      % endif
      tile.input.channel, tile.output.channel${"," if y_data_size_byte != 32 else ""}
      % if y_data_size_byte != 32:
      ${FLAG_RELU}, ${FLAG_BATCHNORM}
      % endif
      );
  % else:
      tile.addr.input,
      im2col,
      % if has_bias:
      tile.addr.bias,
      % else:
      NULL,
      % endif
      tile.addr.output,
      tile.addr.weights,
      % if flag_DW == 1:
      pwt_buffer,
      % endif
      % if FLAG_BATCHNORM == 1:
      tile.addr.scale, tile.addr.bias,
      % else:
      NULL, NULL,
      % endif
      ${out_mul}, ${out_shift},
      tile.input.width, tile.input.height, tile.input.channel,
      tile.output.width, tile.output.height, tile.output.channel,
      ${fs2}, ${fs1},
      tile.padding.top, tile.padding.bottom, tile.padding.left, tile.padding.right, ${stride}, ${stride},
      ${FLAG_RELU}, ${FLAG_BATCHNORM}
      );
  % endif
}

typedef struct ConvolutionArgs {
  Layer tile;
  void * im2col;
  void * pwt_buffer;
} ConvolutionArgs;

static void convolution(void * args) {
  ConvolutionArgs * convolutionArgs = (ConvolutionArgs *)args;
  kernel(convolutionArgs->tile, convolutionArgs->im2col, convolutionArgs->pwt_buffer);
}

GAP_L2_DATA uint8_t l2_W[${weights_dimensions}] = {${weights_vectors}};

void __attribute__ ((noinline)) ${func_name}_internal(
  void *arg 
) {

  unsigned int *real_args = (unsigned int *) arg;
  void * l2_x = (void *) real_args[0];
  void * l2_y = (void *) real_args[1];
  //////////////////////////////////////////////////////////////////////////
  // arguments assigning: keeping same interface between L2 and L3 memory //
  //////////////////////////////////////////////////////////////////////////
  //unsigned int l2_x_2 =(unsigned int)  real_arg[4];
  // DA dichiarare all'interno, ma cosa succede con il multicore??
  unsigned int l1_buffer = pi_cl_l1_malloc(NULL, ${buffer_l1_all}); //Forse manca l'im2col

  const Layer layer = {
    .addr = {
      .input = l2_x,
      .weights = l2_W,
      % if FLAG_BATCHNORM == 1:
      .scale = l2_W + ${l2_off_k},
      .bias = l2_W+${l2_off_lambda},
      % endif
      % if has_bias == 1:
      .bias = l2_W+${l2_off_bias},
      % endif
      .output = l2_y
    },
    .input = {
      .width = ${x_w},
      .channel = ${nif*g}
    },
    .output = {
      .width = ${y_w},
      .channel = ${int(nof*factor)}
    },
    .padding = {
        .top    = ${padding_top},
        .right  = ${padding_right},
        .bottom = ${padding_bottom},
        .left   = ${padding_left}
    }
  };

  DoubleBuffer db_input = {
    .addrs = { l1_buffer + ${l1_x_offset}, l1_buffer + ${l1_x_offset} + ${x_tile_size_byte} },
    .index = 0
  };
  DoubleBuffer db_output = {
    .addrs = { l1_buffer + ${l1_y_offset}, l1_buffer + ${l1_y_offset} + ${y_tile_size_byte} },
    .index = 0
  };
  DoubleBuffer db_weights = {
    .addrs = { l1_buffer + ${l1_W_offset}, l1_buffer + ${l1_W_offset} + ${W_tile_size_byte + k_tile_size_byte_transfer + lambda_tile_size_byte_transfer} },
    .index = 0
  };

  pi_team_config_offload(NUM_CORES);

  DmaTransfer transfer = dma_transfer_create();

  Layer tile_prev;
  TileIndex index_prev = { .height = 0, .width = 0, .input_channel = 0, .output_channel = 0 };
  TileIndex index = { .height = 0, .width = 0, .input_channel = 0, .output_channel = 0 };

  int is_input_load = 1, is_weights_load = 1;

  void * im2col = l1_buffer + ${buffer_l1_all};
  void * pwt_buffer = \
% if flag_DW == 1:
im2col + ${im2col_dim};
% else:
NULL;
% endif

% if flag_DW == 1:
  const int total_tiles = index_end.output_channel * index_end.height * index_end.width;
% else:
  const int total_tiles = index_end.output_channel * index_end.input_channel * index_end.height * index_end.width;
% endif

  // tile loop nest
  for(int iter=0; iter < total_tiles; iter++) {
    Address addr = {
      .input = double_buffer_get_addr(db_input),
      .weights = double_buffer_get_addr(db_weights),
      % if FLAG_BATCHNORM == 1:
      .scale = double_buffer_get_addr(db_weights) + ${W_tile_size_byte},
      .bias = double_buffer_get_addr(db_weights) + ${W_tile_size_byte} + ${k_tile_size_byte_transfer},
      % endif
      % if has_bias == 1:
      .bias = l1_buffer + ${l1_b_offset} + index.output_channel * ${bias_tile_size_byte},
      % endif
      .output = double_buffer_get_addr(db_output)
    };
    Layer tile = tile_create(index, index_end, body, border, layer, addr);

    % if has_bias == 1:
    if (iter == 0) {
      load_bias_async(tile, layer);
    }
    % endif

    if (is_input_load) {
      load_input_async(tile, body, layer, index);
    }
    if (is_weights_load) {
      load_weights_async(tile, body, layer, index);
    }

    ConvolutionArgs convolutionArgs = {
      .tile = tile,
      .im2col = im2col,
      .pwt_buffer = pwt_buffer
    };

    dma_transfer_wait(transfer);

    if (iter > 0) {
      pi_team_offload_wait();
    }

    pi_team_offload_preset(convolution, &convolutionArgs);

    if (iter > 0) {
      store_output_async(tile_prev, body, layer, index_prev);
    }

    tile_prev = tile;
    index_prev = index;
    % if flag_DW == 0:
    index = tile_index_get_next(index, index_end);
    % else:
    index = tile_index_get_next_dw(index, index_end);
    % endif

    is_input_load = index.input_channel!=index_prev.input_channel || index.width!=index_prev.width || index.height!=index_prev.height;
    is_weights_load = index.input_channel!=index_prev.input_channel || index.output_channel!=index_prev.output_channel;

    if (is_input_load) {
      double_buffer_increment(&db_input);
    }
    if (is_weights_load) {
      double_buffer_increment(&db_weights);
    }
    double_buffer_increment(&db_output);
  }

  pi_team_offload_wait();
  store_output_async(tile_prev, body, layer, index_prev);

  dma_transfer_wait(transfer);
  pi_cl_l1_free(NULL, l1_buffer, ${buffer_l1_all});
}

void __attribute__ ((noinline)) ${func_name}(
  void *l2_x, void *l2_y
)
{
    pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_FC, 50000000);
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_CL, 50000000);
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_PERIPH, 50000000);
  pi_time_wait_us(10000);

    // Sono un layer interno!
    uint32_t errors = 0;
    uint32_t core_id = pi_core_id(), cluster_id = pi_cluster_id();
    printf("[%d %d] Hello World!\n", cluster_id, core_id);


  struct pi_device cluster_dev = {0};
  struct pi_cluster_conf conf;
  struct pi_cluster_task cluster_task = {0};

  // First open the cluster
  pi_cluster_conf_init(&conf);
  conf.id=0;
  unsigned int args[2];
  args[0] = (unsigned int) l2_x;
  args[1] = (unsigned int) l2_y;
  // open cluster...
  pi_cluster_task(&cluster_task, ${func_name}_internal, args);
  pi_open_from_conf(&cluster_dev, &conf);
  if (pi_cluster_open(&cluster_dev))
    return;
  // Then offload an entry point, this will get executed on the cluster controller
#ifndef TARGET_CHIP_FAMILY_GAP9
  cluster_task.stack_size = 3500;
#endif
  cluster_task.slave_stack_size = 3400;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  pi_cluster_close(&cluster_dev);
  return 0;

}