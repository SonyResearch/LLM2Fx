from abc import ABC, abstractmethod
from numbers import Number
from typing import Callable, Optional, Tuple, cast
import torch
import torch.nn.functional as F
from loguru import logger
from torch import BoolTensor, FloatTensor, LongTensor, Tensor
from transformers import PreTrainedTokenizer
from .utils import is_number


class AbstractNTLoss(ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        activated_vocab: int,
        digit_level: bool = True,
        reweigh: bool = True,
    ):
        """
        NTL constructor.

        Args:
            tokenizer: Standard HF tokenizer.
            digit_level: Whether to ensure only digits are considered number tokens,
                stabilizing training with NTL. Defaults to True. Used for most
                experiments in the ICML paper.
            reweigh: Whether to scale the NTL using the logit weight on
                number tokens. Defaults to True.
                NOTE: The ICML paper does *not* use this option which can lead to
                incorrect loss if most mass is placed outside of the number tokens.

        """
        super().__init__()
        self.tokenizer = tokenizer
        self.activated_vocab = activated_vocab
        self.digit_level = digit_level
        self.reweigh = reweigh

        self.setup_number_tokens()

        self.max_dist = torch.tensor(0.0)

    def setup_number_tokens(self):
        """Setting up attributes needed by NT loss"""

        # Add digits to vocab if not there yet.
        vocab = self.tokenizer.get_vocab()
        self.number_values: FloatTensor = torch.full((self.activated_vocab,), float("nan"))
        # Try to convert each token to a float after stripping the space prefix
        for token, id in vocab.items():
            if id >= self.activated_vocab:
                continue
            if is_number(token, finite=True):
                if self.digit_level:
                    # NOTE: This check ensures number token value only occurs for digits, not for multi-digit numbers (123)
                    # This stabilizes training with NTL. Can be altered though, see paper experiments.
                    # Excludes tokens that are numbers in other languages like ႘ and tokens with space pre-/postfix like ` 2`.
                    if token.isascii() and -1 <= float(token) <= 9 and len(token) == 1:
                        self.number_values[id] = float(token)
                else:
                    self.number_values[id] = float(token)

        self.is_number_token = ~torch.isnan(self.number_values)
        if self.is_number_token.sum() == len(self.is_number_token):
            raise ValueError("At least one token needs to be not a number, otherwise `ignore_index` cannot be set up safely")
        self.nan_id = torch.where(~self.is_number_token)[0][0].item()
        self.number_values_dense = self.number_values[self.is_number_token]

        if self.digit_level and (num_nts := len(self.number_values_dense)) != 10:
            logger.error(
                f"You requested digit-level but {num_nts} number tokens were identified: {self.number_values_dense}"
            )

    @abstractmethod
    def forward(
        self,
        logits: FloatTensor,
        labels: LongTensor,
        loss_weights: Optional[Tensor] = None,
        reduction: str = "mean",
    ) -> Tensor: ...

    def __call__(self, *args, **kwargs):
        """Alias to self.forward"""
        return self.forward(*args, **kwargs)

    def reweigh_fn(
        self,
        logits: Tensor,
        loss: Tensor,
        number_token_positions: Tensor,
    ) -> Tensor:
        """
        Scale the NT loss element-wise using the logit weight on number tokens.
        NOTE: This reweighing ensures that if ground truth is a number token
            but most probability mass is on text tokens, the loss will be *higher*
            than the worst possible number token. This is an edge case in practice.

        Args:
            logits: 3D Tensor of shape BS x T x V.
            loss: 1D Tensor over all number tokens in batch.
            number_token_positions: 2D Tensor of shape BS x T indicating for which tokens
                the NT loss was computed.

        Returns:
            A 1D Tensor over all number tokens in batch with the scaled NT losses.
        """

        # Take softmax over logits of all tokens in vocab and compute NT logit weight
        softmax_probs_all = F.softmax(logits, dim=-1)
        nt_logit_weight = torch.sum(
            softmax_probs_all[:, :, self.is_number_token], dim=-1
        )[number_token_positions]

        # Apply weights for NTL element-wise
        loss *= nt_logit_weight

        # Apply regularization
        # NOTE: We could consider reweighing here with the max for that label token
        # rather than the global max
        loss += (
            1.01
            * self.max_dist.to(dtype=loss.dtype, device=loss.device)
            * (1 - nt_logit_weight)
        )

        return loss

    @staticmethod
    def _validate_inputs(
        logits: FloatTensor,
        labels: Optional[LongTensor],
        loss_weights: Optional[Tensor],
    ):
        """Private method to perform size and type checks."""
        if (td := len(logits.shape)) != 3 or (ne := logits.numel()) == 0:
            raise ValueError(
                f"Logits have to be non-empty 3D Tensor, not {td}D with {ne} elements"
            )
        if not torch.is_floating_point(logits):
            raise TypeError("Logits have to be FloatTensor.")
        if labels is None:
            return
        if (b := labels.shape) != (a := logits.shape[:-1]):
            raise ValueError(
                f"Logit and label sizes of first 2 dims have to match: {a} vs {b}"
            )

        if (td := len(labels.shape)) != 2 or (ne := labels.numel()) == 0:
            raise ValueError(
                f"Labels have to be non-empty 2D Tensor, not {td}D with {ne} elements"
            )
        if loss_weights is not None:
            if loss_weights.shape != labels.shape:
                raise ValueError(
                    "Loss mask has to be 2D Tensor of same shape as labels."
                )
            if torch.any(loss_weights < 0):
                raise ValueError("loss_mask must be ≥ 0.")

    def _prepare_number_token_targets(
        self, labels: LongTensor, loss_weights: Optional[Tensor], ignore_index: int
    ) -> Tuple[FloatTensor, Tensor]:
        """
        Prepare number-token targets and masks.

        Args:
            labels: 2D Tensor of shape BS x T.
            loss_weights: Optional 2D Tensor of shape BS x T with loss weight for each token.
            ignore_index: Label ID to ignore. Defaults to -100.

        Returns:
            y: 2D Float Tensor of shape BS x T with target numeric values (NaN for non-number tokens).
            loss_weight: 1D Tensor with a potentially individual loss weight for each number token position.
        """
        labels = cast(LongTensor, labels.masked_fill(labels == ignore_index, self.nan_id))
        # Create a mask to filter out non-digit tokens
        device = labels.device
        self.number_values = self.number_values.to(device)
        y = self.number_values[labels]
        number_token_positions = ~torch.isnan(y)
        loss_weights = (
            loss_weights[number_token_positions]
            if loss_weights is not None
            else torch.ones_like(labels, device=labels.device)[number_token_positions]
        )
        return cast(FloatTensor, y), loss_weights

    @staticmethod
    def _apply_reduction(
        loss: Tensor,
        reduction: str,
        loss_weights: Tensor,
        number_token_positions: Tensor,
        logits: Tensor,
    ) -> Tensor:
        """
        Applies the specified reduction type to the calculated loss.

        This method handles 3 types of reduction: "mean", "sum", and "none".
        For "mean" and "sum", it applies weighting using `loss_weights`.
        For "none", it reshapes the loss back to the original batch and sequence
        dimensions.

        Args:
            loss: 1D Tensor containing the loss for each number token in the batch.
            reduction: The reduction method ("mean", "sum", or "none").
            loss_weights: 1D Tensor with a loss weight for each number token.
            number_token_positions: 2D boolean tensor of shape BS x T indicating
                the positions of number tokens.
            logits: 3D Tensor of shape BS x T x V, used to get the original shape
                for the "none" reduction.

        Returns:
            A Tensor representing the reduced loss:
                - 0D tensor if `reduction` is "mean" or "sum".
                - 2D Tensor of shape BS x T if `reduction` is "none".
        """
        if reduction == "mean":
            # Mean pooling (weighted by loss mask)
            loss = torch.dot(
                loss.flatten(), loss_weights.flatten()
            ) / loss_weights.sum().clamp_min(torch.finfo(loss.dtype).eps)
        elif reduction == "sum":
            loss = torch.dot(loss.flatten(), loss_weights.flatten())
        elif reduction == "none":
            # Cast loss for number tokens back to Tensor of size BS x T
            loss_ = torch.zeros(number_token_positions.numel()).to(loss.device)
            loss_[number_token_positions.view(-1)] = loss * loss_weights
            bs, seq_len, _ = logits.size()
            loss = loss_.view(bs, seq_len)

            assert torch.sum(loss[~number_token_positions]) == 0, (
                "NumberTokenLoss computed for non-digit tokens!"
            )

        else:
            raise ValueError(f"{reduction} is not a valid value for reduction")

        return loss


class NTLossDotProduct(AbstractNTLoss):
    """Class for NT losses that produce a token-wise numerical output."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        digit_level: bool = True,
        reweigh: bool = True,
        loss_function: Callable = F.mse_loss,
    ):
        """
        Referred to as NTL-L_p in the paper.

        Args:
            tokenizer: NTLTokenizer with necessary attributes like is_number_token etc.
            digit_level: Whether to ensure only digits are considered number tokens,
                stabilizing training with NTL. Defaults to True. Used for most
                experiments in the ICML paper.
            reweigh: Whether to scale the NTL using the logit weight on
                number tokens. Defaults to True.
                NOTE: The ICML paper does *not* use this option which can lead to
                incorrect loss if most mass is placed outside of the number tokens.
            loss_function: Function to apply on the delta between the ground truth number
                and the obtained dot product (nt-probs * token-values). Defaults to
                MSE, but MAE, Huber etc are also compatible.
        """
        super().__init__(
            tokenizer=tokenizer,
            digit_level=digit_level,
            reweigh=reweigh,
        )
        self.loss_function = loss_function
        self.setup_max_dist()

    def setup_max_dist(self):
        """
        Set up the maximum distance between the number tokens based on the selected loss function.
        """

        # Extract the number token values and get the minimum and maximum
        vals = self.number_values_dense.unsqueeze(0)
        max_val = vals.max()
        min_val = vals.min()

        # Compute the largest value the loss function used in NT loss computation can get
        # Make sure to account for possibility of asymmetrical loss function
        self.max_dist = torch.maximum(
            torch.abs(self.loss_function(min_val, max_val)),
            torch.abs(self.loss_function(max_val, min_val)),
        )

    def predict_numbers(self, logits: FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
        """
        Calculates token-level numerical prediction.
        NOTE: This calculates numerical predictions for *all* tokens, not just where
        label is a number token.

        Args:
            logits: 3D FloatTensor of shape BS x T x V.

        Returns:
            yhat: 2D FloatTensor BS x T containing numerical predictions.
            nt_mass: 2D FloatTensor BS x T with the cumulated mass assigned to all number tokens.
        """
        self._validate_inputs(logits, labels=None, loss_weights=None)

        # Calculate the token-level predictions
        yhat = self._get_dot_product(logits=logits)

        probs_all = F.softmax(logits, dim=-1)
        probs_nt = probs_all[:, :, self.is_number_token]
        nt_mass = probs_nt.sum(dim=-1)
        return yhat, cast(FloatTensor, nt_mass)

    def _get_dot_product(
        self, logits: FloatTensor, number_token_positions: Optional[BoolTensor] = None
    ) -> FloatTensor:
        """
        Applies dot product of number token values and their predicted probabilites.

        Args:
            logits: 3D FloatTensor of shape BS x T x V.
            number_token_positions: Optional 2D BoolTensor (BS x T) containing locations
                of number tokens.

        Returns:
            If `number_token_positions` is None, 2D FloatTensor of shape BS x T.
            Otherwise, 1D FloatTensor containing the predictions for the number tokens.
        """
        # apply softmax solely over the number token indices
        nt_logits = logits[:, :, self.is_number_token]
        softmax_probs = F.softmax(nt_logits, dim=-1)
        values = self.number_values_dense.to(device=logits.device, dtype=logits.dtype)

        # compute the weighted average of number tokens
        if number_token_positions is None:
            # Calculate for all tokens
            yhat = torch.sum(softmax_probs * values, dim=-1)
        else:
            # Calculate selectively where labels are number tokens
            yhat = torch.sum(softmax_probs[number_token_positions] * values, dim=-1)
        return cast(FloatTensor, yhat)

    def forward(
        self,
        logits: FloatTensor,
        labels: LongTensor,
        loss_weights: Optional[Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> Tensor:
        """
        Computes the NTL based on the dot product between token values and their probs.

        Args:
            logits: 3D Tensor of shape BS x T x V.
            labels: 2D Tensor of shape BS x T.
            loss_weights: 2D Optional tensor of BS x T with token-wise loss weights.
            reduction: Optional string specifying the reduction to apply to the
                output. Defaults to "mean", options are "mean", "sum", "none".
            ignore_index: The token ID to ignore in the labels. Defaults to -100.

        Returns:
            Loss tensor
                OD if reduction=="mean"|"sum"
                BS x T if reduction=="none"
        """
        self._validate_inputs(logits, labels, loss_weights)

        y, loss_weights = self._prepare_number_token_targets(
            labels, loss_weights, ignore_index
        )
        loss_weights = loss_weights.to(logits.dtype)
        number_token_positions = cast(BoolTensor, ~torch.isnan(y))

        # If no digit tokens in batch, or total of the relevant loss weights is zero, no need for upcoming calculations
        if not number_token_positions.any() or not loss_weights.any():
            if (reduction == "mean") | (reduction == "sum"):
                loss = torch.tensor(0, dtype=logits.dtype, device=labels.device)
            elif reduction == "none":
                loss = torch.zeros_like(
                    labels, dtype=logits.dtype, device=labels.device
                )
            else:
                raise ValueError(f"{reduction} is not a valid value for reduction")
            return loss
        yhat = self._get_dot_product(
            logits=logits, number_token_positions=number_token_positions
        )
        # Apply specified loss function to y and yhat
        loss = self.loss_function(yhat, y[number_token_positions], reduction="none")
        # If reweigh: compute weights for NTL based on logits
        if self.reweigh:
            loss = self.reweigh_fn(
                logits=logits, loss=loss, number_token_positions=number_token_positions
            )
        loss = self._apply_reduction(
            loss=loss,
            reduction=reduction,
            loss_weights=loss_weights,
            number_token_positions=number_token_positions,
            logits=logits,
        )
        return loss


class NTLoss(AbstractNTLoss):
    """Class for Wasserstein-based NTLoss. This is the default in the ICML paper."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        activated_vocab: int,
        digit_level: bool = True,
        reweigh: bool = True,
        squash_factor: Optional[float] = None,
    ):
        """
        NTL constructor for the Wasserstein-based NTLoss.

        Args:
            tokenizer: Any HuggingFace tokenizer.
            digit_level: Whether to ensure only digits are considered number tokens,
                stabilizing training with NTL. Defaults to True. Used for most
                experiments in the ICML paper.
            reweigh: Whether to scale the NTL using the logit weight on
                number tokens. Defaults to True.
                NOTE: The ICML paper does *not* use this option which can lead to
                incorrect loss if most mass is placed outside of the number tokens.
            squash_factor: The optional squashing factor for the NTL. If provided,
                this number denotes the factor by which predicting the largest number
                token is worse than predicting the closest incorrect number token.
                E.g., with digit-level tokenization this factor is 9. Setting this
                to 1 will recover cross entropy. This argument is intended to handle
                irregular vocabs with large numerical token values.
        """
        super().__init__(
            tokenizer=tokenizer,
            activated_vocab=activated_vocab,
            digit_level=digit_level,
            reweigh=reweigh,
        )

        self.squash_factor = squash_factor
        self.setup_distance_lookup(squash_factor)

    def setup_distance_lookup(
        self,
        squash_factor: Optional[float] = None,
    ) -> None:
        """
        Set up a lookup table for the distances between the number tokens.
        Use squash_factor to control by what factor the farthest number token is worse than the closest, incorrect number token.
        If not squash_factor is not set: with 10 number tokens (0-9), the squashing factor is 9.
        NOTE: With a squashing factor of 1, this basically collapses to cross entropy.

        Args:
            squash_factor: The optional squashing factor used.
        """

        # Get token ids for number tokens
        num_ids = torch.nonzero(self.is_number_token, as_tuple=True)[0]
        # Create mapping from number token ids to their index in order of appearance in vocab:
        # e.g. token "3" -> id 519 -> dist_idx 1, then abs dist to 3 for other NT values will be found in row/column 1
        vocab_to_dist_idx = torch.full((len(self.tokenizer),), -1, dtype=torch.long)
        # Use arange to ensure order of appearance
        vocab_to_dist_idx[num_ids] = torch.arange(num_ids.size(0), dtype=torch.long)

        # Build NxN abs-diff matrix
        vals = self.number_values_dense.unsqueeze(0)  # (1 x N)
        diff = torch.abs(vals - vals.t())  # (N x N)

        if isinstance(squash_factor, Number):
            assert squash_factor > 1, (
                f"The squash factor can't be equal to or smaller than 1, please use a different squashing factor than {squash_factor}"
            )

            # Mask out zeros to find the smallest nonzero diff
            inf = torch.finfo(diff.dtype).max
            diff_nonzero = diff.masked_fill(diff == 0, inf)
            global_min_nz = diff_nonzero.min()
            # Find largest diff
            global_max = diff.max()

            # Compute scaling factor based on indicated squash factor
            scale = (squash_factor - 1) / (global_max - global_min_nz)
            # Scale the absolute differences using scaling factor
            lookup = 1 + (diff - global_min_nz) * scale
            lookup[diff == 0] = 0.0

        else:
            lookup = diff

        self.vocab_to_dist_idx = vocab_to_dist_idx
        self.dist_lookup = lookup
        self.max_dist = lookup.max()

    def forward(
        self,
        logits: FloatTensor,
        labels: LongTensor,
        loss_weights: Optional[Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> Tensor:
        """
        Computes the NTL.

        Args:
            logits: 3D Tensor of shape BS x T x V.
            labels: 2D Tensor of shape BS x T.
            loss_weights: Optional 2D tensor of BS x T with token-specific weights.
            reduction: Optional string specifying the reduction to apply to the
                output. Defaults to "mean", options are "mean", "sum", "none".
            ignore_index: The token ID to ignore in the labels. Defaults to -100.

        Returns:
            Loss tensor
                OD if reduction=="mean"|"sum"
                BS x T if reduction=="none"

        """
        self._validate_inputs(logits, labels, loss_weights)

        y, loss_weights = self._prepare_number_token_targets(
            labels, loss_weights, ignore_index
        )
        loss_weights = loss_weights.to(logits.dtype)
        number_token_positions = ~torch.isnan(y)

        # If no digit tokens in batch, or total of the relevant loss_weights is zero, no need for upcoming calculations
        if not number_token_positions.any() or not loss_weights.any():
            if (reduction == "mean") | (reduction == "sum"):
                loss = torch.tensor(0, dtype=logits.dtype, device=labels.device)
            elif reduction == "none":
                loss = torch.zeros_like(
                    labels, dtype=logits.dtype, device=labels.device
                )
            else:
                raise ValueError(f"{reduction} is not a valid value for reduction")

            return loss

        # apply softmax and get number labels
        nt_logits = logits[:, :, self.is_number_token]
        softmax_probs = F.softmax(nt_logits, dim=-1)

        # get distance between the true numbers and all possible number values from lookup table
        abs_diff = self.dist_lookup.to(dtype=logits.dtype, device=logits.device)[
            self.vocab_to_dist_idx.to(device=labels.device)[
                labels[number_token_positions]
            ]
        ]

        # loss is the absolute difference weighted by the softmax probs
        loss = (abs_diff * softmax_probs[number_token_positions]).sum(dim=-1)

        # If reweigh: compute weights for NTL based on logits
        if self.reweigh:
            loss = self.reweigh_fn(
                logits=logits, loss=loss, number_token_positions=number_token_positions
            )

        loss = self._apply_reduction(
            loss=loss,
            reduction=reduction,
            loss_weights=loss_weights,
            number_token_positions=number_token_positions,
            logits=logits,
        )

        return loss
