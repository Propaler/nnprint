from nnprint.visualizations.base import BaseVisualization
from nnprint.utils import color_palette, map_to_color


class NormScaleGrid(BaseVisualization):
    """Filter norm visualization grid"""

    def print():

        square_size = 16
        inner_square_margin = 1
        linewidth = 1
        extra_padding_bottom = 10
        extra_margin_left = 30
        max_line_squares = 16
        unique_colors = set()

        colors = color_palette()
        num_colors = len(colors)

        initial_point = (100, 16)
        cur_point = initial_point  # TODO must be defined by default or params values

        # layer_names = list(list(self._model.modules())[0]._modules.keys())
        layer_names = self._model.get_layers_info()
        # print(layer_names)

        layer_id = 0
        for m in list(model.modules())[1:]:
            if isinstance(m, nn.Conv2d):
                weight_copy = m.weight.data.clone().numpy()

                if importance_criteria == "l1":
                    abs_weight = np.abs(weight_copy)
                    norm = np.sum(abs_weight, axis=(1, 2, 3))
                elif importance_criteria == "l2":
                    squared_weight = np.square(weight_copy)
                    norm = np.sum(squared_weight, axis=(1, 2, 3))
                elif importance_criteria == "gm":
                    weight_copy = weight_copy.reshape(weight_copy.shape[0], -1)
                    similarity_matrix = distance.cdist(
                        weight_copy, weight_copy, "euclidean"
                    )
                    norm = np.sum(np.abs(similarity_matrix), axis=0)
                else:
                    raise NotImplementedError(
                        f"There is no support for {importance_criteria} criteria."
                    )

                norm_map = map_to_color(norm)

                self.draw_text(self._base, cur_point, layer_names[layer_id])
                out_features = m.weight.data.shape[0]

                for i in range(out_features):
                    if i > 0 and i % max_line_squares == 0:
                        cur_point = (
                            cur_point[0]
                            - (square_size + inner_square_margin + linewidth)
                            * max_line_squares,
                            cur_point[1]
                            + square_size
                            + inner_square_margin
                            + linewidth,
                        )
                    colour_index = norm_map[i] % num_colors
                    colour = colors[colour_index]
                    unique_colors.add((colour, colour_index))
                    cur_point = self.draw_square(self._base, cur_point, fill=colour)
                    cur_point = (
                        cur_point[0] + inner_square_margin,
                        cur_point[1] - square_size - inner_square_margin,
                    )
            elif isinstance(m, nn.Linear):
                weight_copy = m.weight.data.clone().numpy()

                if importance_criteria == "l1":
                    abs_weight = np.abs(weight_copy)
                    norm = np.sum(abs_weight, axis=(1))
                elif importance_criteria == "l2":
                    squared_weight = np.square(weight_copy)
                    norm = np.sum(squared_weight, axis=(1))
                elif importance_criteria == "gm":
                    similarity_matrix = distance.cdist(
                        weight_copy, weight_copy, "euclidean"
                    )
                    norm = np.sum(np.abs(similarity_matrix), axis=0)
                else:
                    raise NotImplementedError(
                        f"There is no support for {importance_criteria} criteria."
                    )

                norm_map = map_to_color(norm)

                self.draw_text(self._base, cur_point, layer_names[layer_id])
                out_features = m.weight.data.shape[0]

                for i in range(out_features):
                    if i > 0 and i % max_line_squares == 0:
                        cur_point = (
                            cur_point[0]
                            - (square_size + inner_square_margin + linewidth)
                            * max_line_squares,
                            cur_point[1]
                            + square_size
                            + inner_square_margin
                            + linewidth,
                        )
                    colour_index = norm_map[i] % num_colors
                    colour = colors[colour_index]
                    unique_colors.add((colour, colour_index))
                    cur_point = self.draw_square(self._base, cur_point, fill=colour)
                    cur_point = (
                        cur_point[0] + inner_square_margin,
                        cur_point[1] - square_size - inner_square_margin,
                    )

            # add extra padding between layers
            cur_point = (
                initial_point[0],
                cur_point[1]
                + square_size
                + inner_square_margin
                + linewidth
                + extra_padding_bottom,
            )
            layer_id += 1

        # adding legend
        # FIXME should be possible set legend position

        legend_colors = list(unique_colors)

        # sort by norm
        legend_colors.sort(key=lambda item: item[1])

        # print(legend_colors)
        cur_point = (
            initial_point[0]
            + square_size * (max_line_squares + linewidth + inner_square_margin)
            + extra_margin_left,
            initial_point[1],
        )
        for i, (colour, aprox_norm) in enumerate(legend_colors):
            if i == 0:
                self.draw_text(
                    self._base,
                    (cur_point[0] - square_size - inner_square_margin, cur_point[1],),
                    "lowest",
                    position="right",
                )
            elif i == len(legend_colors) - 1:
                self.draw_text(
                    self._base,
                    (cur_point[0] - square_size - inner_square_margin, cur_point[1],),
                    "highest",
                    position="right",
                )
            cur_point = self.draw_square(self._base, cur_point, fill=colour)
            cur_point = (
                cur_point[0] - square_size - inner_square_margin,
                cur_point[1] + inner_square_margin + linewidth,
            )

        return self._base

    def save(path):
        self._base.save(path)
