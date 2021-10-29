# Local Neural Transformations (LNT) - a self-supervised method for
# anomalous region detection in time series
# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

class EarlyStoppingCriterion:

    def __init__(self):
        pass

    def inform_value(self, new_value):
        pass

    def is_stop_required(self):
        pass


    @staticmethod
    def get_from_config(config):

        if hasattr(config, 'early_stopping'):

            assert hasattr(config.early_stopping, 'type')



        else:
            return None


class NoImprovementStoppingCriterion(EarlyStoppingCriterion):

    def __init__(self, no_improvement_for):
        super(NoImprovementStoppingCriterion, self).__init__()

        self._best_value = None
        self._no_improvement_for = no_improvement_for
        self._counter = 0

    def inform_value(self, new_value):

        if self._best_value is None or new_value < self._best_value:
            self._best_value = new_value
            self._counter = 0
        else:
            self._counter += 1

    def is_stop_required(self):
        return self._counter >= self._no_improvement_for

