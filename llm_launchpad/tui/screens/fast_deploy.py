"""Model-first Deploy flow over live compute availability."""

from __future__ import annotations

from dataclasses import dataclass

from textual.actions import SkipAction
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Footer, OptionList, Select, Static
from textual.widgets.option_list import Option

from ...core.compute_availability import (
    display_gpu_type,
    load_compute_availability,
    plans_for_compute_profile,
)
from ...core.quick_deploy import (
    QuickDeployCatalogInfo,
    QuickDeployModel,
    QuickDeployProfile,
    format_context_length,
    format_hourly_cost,
    get_quick_deploy_catalog_info,
    list_quick_deploy_models,
    quick_deploy_recipe,
    resolve_quick_deploy_plans,
)
from ...protocol.enums import ComputeProvider
from ...protocol.models import (
    ComputeAvailabilitySnapshot,
    ComputeConfiguration,
    InferencePlan,
)
from ..responsive import ViewportProfile, WidthMode
from .copy_enabled import CopyEnabledScreen


class FastDeployAvailabilityLoaded(Message):
    """Live compute availability loaded for a fast deploy model."""

    def __init__(
        self,
        snapshot: ComputeAvailabilitySnapshot,
        *,
        request_id: int | None = None,
        purpose: str = "infra",
    ) -> None:
        super().__init__()
        self.snapshot = snapshot
        self.request_id = request_id
        self.purpose = purpose


class FastDeployAvailabilityFailed(Message):
    """Live compute availability could not be loaded."""

    def __init__(
        self,
        error: str,
        *,
        request_id: int | None = None,
        purpose: str = "infra",
    ) -> None:
        super().__init__()
        self.error = error
        self.request_id = request_id
        self.purpose = purpose


@dataclass(frozen=True)
class _InfraRow:
    """One deployable (profile, plan) pairing behind a fast-deploy row."""

    profile: QuickDeployProfile
    plan: InferencePlan
    configuration: ComputeConfiguration
    alternative_plans: tuple[InferencePlan, ...]


def _escape_markup(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _clip(value: str, width: int) -> str:
    text = (value or "").strip()
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return f"{text[:width - 3]}..."


def _format_price(value: float | None, *, estimate: bool = False) -> str:
    if value is None:
        return "price n/a"
    prefix = "~" if estimate else ""
    return f"{prefix}${value:.2f}/hr"


def _monthly_label(value: float | None) -> str:
    if value is None:
        return "monthly est. n/a"
    return f"~${value:,.0f}/mo"


def _compact_quant_label(quant: str) -> str:
    cleaned = quant.strip().strip("()")
    if not cleaned:
        return ""
    return (
        cleaned.replace("UD-", "")
        .replace("_K_XL", "XL")
        .replace("_K_M", "M")
        .replace("_K_S", "S")
        .replace("_", "")
        .replace("-", "")
    )


def _quant_markup(profile: QuickDeployProfile) -> str:
    quant = _compact_quant_label(profile.quant)
    if not quant:
        return ""
    if quant.casefold().startswith("q2"):
        return f"[bold #7bf168]{_escape_markup(quant)}[/]"
    return f"[dim]{_escape_markup(quant)}[/dim]"


def _format_aa_index(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _capacity_label(count: int, adjective: str, color: str) -> str:
    noun = "placement" if count == 1 else "placements"
    return f"[{color}]{count} {adjective} {noun}[/]"


def _availability_label(configuration: ComputeConfiguration) -> str:
    parts = []
    if configuration.live_placement_count:
        parts.append(_capacity_label(configuration.live_placement_count, "live", "green"))
    if configuration.has_on_demand_capacity:
        parts.append("[yellow]on demand[/yellow]")
    if configuration.spot_placement_count:
        parts.append(_capacity_label(configuration.spot_placement_count, "spot", "magenta"))
    return " + ".join(parts) or "[dim]availability unknown[/dim]"


def _subtitle(info: QuickDeployCatalogInfo) -> str:
    return f"Pick a model · {_escape_markup(info.source_label)}"


def _model_cheapest_price(
    model: QuickDeployModel,
    snapshot: ComputeAvailabilitySnapshot | None = None,
) -> tuple[float | None, bool]:
    """Return the cheapest hourly price and whether it is an estimate."""

    catalog = min(
        (profile.approx_cost_per_hour_usd for profile in model.profiles),
        default=None,
    )
    if snapshot is None:
        return catalog, True
    priced = [
        row
        for row in infra_rows_for_model(model, snapshot)
        if row.plan.quote.price_per_hour_usd is not None
    ]
    if not priced:
        return catalog, True
    best = min(
        priced,
        key=lambda row: row.plan.quote.price_per_hour_usd or float("inf"),
    )
    return best.plan.quote.price_per_hour_usd, best.plan.quote.is_estimate


def _model_cost_label(
    model: QuickDeployModel,
    snapshot: ComputeAvailabilitySnapshot | None = None,
) -> str:
    price, estimate = _model_cheapest_price(model, snapshot)
    if price is None:
        return "price n/a"
    if snapshot is None:
        return format_hourly_cost(price)
    return _format_price(price, estimate=estimate)


def _model_option(
    model: QuickDeployModel,
    width_mode: WidthMode = WidthMode.WIDE,
    snapshot: ComputeAvailabilitySnapshot | None = None,
) -> str:
    cost = _model_cost_label(model, snapshot)
    score = (
        f"AAI {_format_aa_index(model.quality_score)}"
        if model.quality_score is not None
        else "unranked"
    )
    size = (model.profiles[0].model_size_label or "").strip()
    metrics = f"{size} · " if size else ""
    if width_mode == WidthMode.MINIMAL:
        return (
            f"  {_escape_markup(_clip(model.display_name, 25))}  "
            f"[dim]{cost}[/dim]"
        )
    if width_mode == WidthMode.COMPACT:
        return (
            f"  {_escape_markup(_clip(model.display_name, 28)):<28} "
            f"[dim]{score} · {cost}[/dim]"
        )
    return (
        f"  {_escape_markup(_clip(model.display_name, 34)):<34} "
        f"[dim]{metrics}{score} · from {cost}[/dim]"
    )


def _unique_quant_labels(model: QuickDeployModel) -> tuple[str, ...]:
    labels: list[str] = []
    seen: set[str] = set()
    for recipe in model.recipes:
        quant = (recipe.quant or "").strip()
        key = quant.casefold()
        if quant and key not in seen:
            seen.add(key)
            labels.append(quant)
    if labels:
        return tuple(labels)
    for profile in model.profiles:
        quant = (profile.quant or "").strip()
        key = quant.casefold()
        if quant and key not in seen:
            seen.add(key)
            labels.append(quant)
    return tuple(labels)


def _quant_options_label(model: QuickDeployModel) -> str:
    labels = _unique_quant_labels(model)
    count = len(labels) or len(model.recipes) or len(model.profiles)
    if labels:
        return f"{count} quant option{'s' if count != 1 else ''}"
    return f"{count} option{'s' if count != 1 else ''}"


def _model_detail(model: QuickDeployModel) -> str:
    score = (
        f"AAI {_format_aa_index(model.quality_score)}"
        if model.quality_score is not None
        else "unranked"
    )
    return (
        f"[bold]{_escape_markup(model.display_name)}[/bold]  "
        f"{_escape_markup(format_context_length(model.max_context_tokens))}\n"
        f"[dim]{score} · {_quant_options_label(model)} · "
        f"pick one to see live infrastructure[/dim]"
    )


def representative_profiles_for_model(
    model: QuickDeployModel,
) -> tuple[QuickDeployProfile, ...]:
    """Return one catalog profile per unique inference recipe."""

    unique: list[QuickDeployProfile] = []
    seen: set[str] = set()
    for profile in model.profiles:
        recipe_id = quick_deploy_recipe(profile).id
        if recipe_id in seen:
            continue
        seen.add(recipe_id)
        unique.append(profile)
    return tuple(unique)


def infra_rows_for_model(
    model: QuickDeployModel,
    snapshot: ComputeAvailabilitySnapshot,
) -> tuple[_InfraRow, ...]:
    """Build unique live placements for a model, cheapest first.

    Rows are one GPU type × quant × provider. Extra regions for the same
    shape stay attached as fulfillment alternatives instead of duplicate
    list entries.
    """

    rows: list[_InfraRow] = []
    seen: set[tuple[str, str, str]] = set()
    for configuration in snapshot.configurations:
        for profile in representative_profiles_for_model(model):
            grouped: dict[str, list[InferencePlan]] = {}
            for plan in plans_for_compute_profile(configuration, profile):
                grouped.setdefault(plan.quote.provider.value, []).append(plan)
            for provider_plans in grouped.values():
                best = provider_plans[0]
                key = (configuration.id, best.recipe.id, best.quote.provider.value)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    _InfraRow(
                        profile=profile,
                        plan=best,
                        configuration=configuration,
                        alternative_plans=tuple(provider_plans),
                    )
                )
    rows.sort(key=_infra_sort_key)
    return tuple(rows)


def _infra_gpu_label(row: _InfraRow) -> str:
    return row.configuration.gpu_type or display_gpu_type(row.plan.quote.gpu_type)


def _infra_option(row: _InfraRow, width_mode: WidthMode = WidthMode.WIDE) -> str:
    quote = row.plan.quote
    gpu = _infra_gpu_label(row)
    price = _format_price(quote.price_per_hour_usd, estimate=quote.is_estimate)
    monthly = _monthly_label(row.plan.estimated_monthly_cost_usd)
    extra = ""
    if len(row.alternative_plans) > 1:
        extra = f" · {len(row.alternative_plans)} placements"
    quant = _quant_markup(row.profile)
    provider = _escape_markup(quote.provider.display_name)
    if width_mode == WidthMode.MINIMAL:
        return (
            f"  {_escape_markup(_clip(gpu, 14))} x{quote.gpu_count} "
            f"{quant} [dim]{provider} {price}[/dim]"
        )
    if width_mode == WidthMode.COMPACT:
        return (
            f"  {_escape_markup(_clip(gpu, 16)):<16} x{quote.gpu_count:<2} "
            f"{quant} {provider} [dim]{price}[/dim]"
        )
    return (
        f"  {_escape_markup(gpu):<20} x{quote.gpu_count:<2} "
        f"{quant} {provider} "
        f"{_availability_label(row.configuration)}  "
        f"[dim]{price} · {monthly}{extra}[/dim]"
    )


def _infra_detail(row: _InfraRow) -> str:
    quote = row.plan.quote
    gpu = _infra_gpu_label(row)
    price = _format_price(quote.price_per_hour_usd, estimate=quote.is_estimate)
    monthly = _monthly_label(row.plan.estimated_monthly_cost_usd)
    region = (quote.region or "").strip() or "provider-managed regions"
    return (
        f"[bold]{_escape_markup(gpu)}[/bold] x{quote.gpu_count}  "
        f"{_escape_markup(quote.provider.display_name)} · {_quant_markup(row.profile)}\n"
        f"[dim]{price} · {monthly} · {region} · {_availability_label(row.configuration)}[/dim]"
    )


def _infra_sort_key(row: _InfraRow) -> tuple[int, float, str, int]:
    price = row.plan.quote.price_per_hour_usd
    return (
        1 if price is None else 0,
        price if price is not None else float("inf"),
        row.plan.quote.gpu_type.casefold(),
        row.plan.quote.gpu_count,
    )


def _gpu_filter_options(snapshot: ComputeAvailabilitySnapshot) -> list[tuple[str, str]]:
    """Return Select options for unique GPU types, cheapest first."""
    options: list[tuple[str, str]] = [("Any GPU", "any")]
    ranked: list[tuple[str, float | None]] = []
    seen: set[str] = set()
    for configuration in snapshot.configurations:
        gpu_type = configuration.gpu_type.strip()
        if not gpu_type or gpu_type in seen:
            continue
        seen.add(gpu_type)
        prices = [
            row.minimum_price_per_hour_usd
            for row in snapshot.configurations
            if row.gpu_type == gpu_type and row.minimum_price_per_hour_usd is not None
        ]
        ranked.append((gpu_type, min(prices) if prices else None))
    ranked.sort(key=lambda item: (1 if item[1] is None else 0, item[1] or 0.0, item[0].casefold()))
    for gpu_type, price in ranked:
        label = gpu_type if price is None else f"{gpu_type} · from {_format_price(price)}"
        options.append((label, gpu_type))
    return options


def _model_fits_gpu_type(
    model: QuickDeployModel,
    snapshot: ComputeAvailabilitySnapshot,
    gpu_type: str,
) -> bool:
    """Return True when the model has at least one fulfillment on ``gpu_type``."""
    configurations = [row for row in snapshot.configurations if row.gpu_type == gpu_type]
    if not configurations:
        return False
    for profile in representative_profiles_for_model(model):
        if any(plans_for_compute_profile(configuration, profile) for configuration in configurations):
            return True
    return False


def _filter_infra_rows(rows: tuple[_InfraRow, ...], gpu_type: str) -> tuple[_InfraRow, ...]:
    if not gpu_type or gpu_type == "any":
        return rows
    return tuple(
        row
        for row in rows
        if row.configuration.gpu_type == gpu_type or row.plan.quote.gpu_type == gpu_type
    )


def _tier_markup(profile: QuickDeployProfile) -> str:
    label = (profile.resource_tier_label or "").strip()
    if not label:
        tier = (profile.resource_tier or "").strip().casefold()
        if tier == "cheap":
            label = "$"
        elif tier == "rtx-pro":
            label = "$$"
        elif tier == "b200":
            label = "$$$"
    if not label:
        return ""
    if label == "$":
        return "[dim]$[/dim]"
    if label == "$$":
        return "[#7bf168]$$[/]"
    if label == "$$$":
        return "[yellow]$$$[/yellow]"
    return f"[dim]{_escape_markup(label)}[/dim]"


def _compact_gpu_shape(profile: QuickDeployProfile) -> str:
    gpu = profile.gpu_type.strip()
    compact = {
        "A100-80GB": "A100",
        "A100-40GB": "A100-40",
        "RTX-PRO-6000": "RTX6000",
    }.get(gpu, gpu.replace("-80GB", ""))
    return f"{compact}x{profile.gpu_count}"


def _fallback_option(profile: QuickDeployProfile) -> str:
    shape = _compact_gpu_shape(profile)
    cost = format_hourly_cost(profile.approx_cost_per_hour_usd)
    return (
        f"  {_escape_markup(_clip(profile.display_name, 34)):<34} "
        f"{_tier_markup(profile)} {_quant_markup(profile)} "
        f"[dim]{shape} · {cost}[/dim]"
    )


def _fallback_detail(profile: QuickDeployProfile) -> str:
    shape = _compact_gpu_shape(profile)
    cost = format_hourly_cost(profile.approx_cost_per_hour_usd)
    return (
        f"[bold]{_escape_markup(profile.display_name)}[/bold]  {_quant_markup(profile)}\n"
        f"[dim]{shape} · {cost} · catalog estimate (availability unavailable)[/dim]"
    )


class FastDeployScreen(CopyEnabledScreen):
    """Pick a catalog model, filter by GPU, then confirm a live placement."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("enter", "choose_selected", "Choose", show=True, priority=True),
        Binding("g", "focus_gpu_filter", "GPU filter", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._catalog_info = get_quick_deploy_catalog_info()
        self._models = list_quick_deploy_models()
        self._model_by_id = {model.id: model for model in self._models}
        self._selected_model: QuickDeployModel | None = None
        self._infra_rows: dict[str, _InfraRow] = {}
        self._fallback_profiles: dict[str, QuickDeployProfile] = {}
        self._availability_inflight = False
        self._availability_request_id = 0
        self._phase = "models"
        self._snapshot: ComputeAvailabilitySnapshot | None = None
        self._gpu_filter = "any"
        self._updating_gpu_filter = False

    def compose(self) -> ComposeResult:
        with Vertical(id="fast-deploy-container"):
            yield Static(
                "[bold #7bf168]Deploy[/]  [dim]Step 1: Pick a model[/dim]",
                id="fast-deploy-title",
            )
            yield Static(_subtitle(self._catalog_info), id="fast-deploy-subtitle")
            yield Static("[dim]GPU filter[/dim]", id="fast-deploy-gpu-label")
            yield Select(
                options=[("Any GPU", "any")],
                value="any",
                allow_blank=False,
                id="fast-deploy-gpu-filter",
            )
            yield Static("[dim]Loading models...[/dim]", id="fast-deploy-status")
            yield OptionList(id="fast-deploy-list")
            yield Static("", id="fast-deploy-detail")
        yield Footer()

    def on_mount(self) -> None:
        self._availability_inflight = False
        self._apply_model_catalog(force=True)
        self.query_one("#fast-deploy-list", OptionList).focus()
        self._load_availability(purpose="filter")

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id != "fast-deploy-list":
            return
        option_id = str(event.option.id)
        if self._phase == "models":
            model = self._model_by_id.get(option_id)
            if model is not None:
                self._update_model_detail(model)
        elif self._phase == "infra":
            row = self._infra_rows.get(option_id)
            if row is not None:
                self._update_infra_detail(row)
        elif self._phase == "fallback":
            profile = self._fallback_profiles.get(option_id)
            if profile is not None:
                self._update_fallback_detail(profile)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "fast-deploy-list":
            self._choose(str(event.option.id))

    def _choose(self, option_id: str) -> None:
        if self._phase == "loading":
            return
        if self._phase == "models":
            self._open_model(option_id)
            return
        if self._availability_inflight:
            return
        if self._phase == "infra":
            row = self._infra_rows.get(option_id)
            if row is not None:
                self.app.push_quick_deploy(  # type: ignore[attr-defined]
                    row.plan,
                    alternative_plans=row.alternative_plans,
                )
            return
        profile = self._fallback_profiles.get(option_id)
        if profile is None:
            return
        plans = resolve_quick_deploy_plans((profile,))
        if plans:
            self.app.push_quick_deploy(  # type: ignore[attr-defined]
                plans[0],
                alternative_plans=plans,
            )
        else:
            self.app.push_quick_deploy(profile)  # type: ignore[attr-defined]

    def _open_model(self, option_id: str) -> None:
        model = self._model_by_id.get(option_id)
        if model is None or self._availability_inflight:
            return
        self._selected_model = model
        self.query_one("#fast-deploy-title", Static).update(
            f"[bold #7bf168]Deploy[/]  "
            f"[dim]{_escape_markup(model.display_name)} · Step 2: Pick infrastructure[/dim]"
        )
        self.query_one("#fast-deploy-status", Static).update(
            "[dim]Checking live infrastructure across connected sources...[/dim]"
        )
        option_list = self.query_one("#fast-deploy-list", OptionList)
        option_list.set_options(
            [Option("  Checking live infrastructure...", disabled=True)]
        )
        self.query_one("#fast-deploy-detail", Static).update("")
        option_list.focus()
        if self._snapshot is not None:
            self._render_infra_from_snapshot(model, self._snapshot)
            return
        self._phase = "loading"
        self._load_availability(purpose="infra")

    def _load_availability(self, *, purpose: str) -> None:
        self._availability_request_id += 1
        request_id = self._availability_request_id
        if purpose == "infra":
            self._availability_inflight = True

        def run_load() -> None:
            self._run_load_availability(request_id, purpose=purpose)

        self.run_worker(
            run_load,
            name="fast-deploy-availability-worker",
            thread=True,
        )

    def _run_load_availability(self, request_id: int, *, purpose: str = "infra") -> None:
        try:
            snapshot = load_compute_availability()
        except Exception as exc:
            self.post_message(
                FastDeployAvailabilityFailed(
                    str(exc),
                    request_id=request_id,
                    purpose=purpose,
                )
            )
            return
        self.post_message(
            FastDeployAvailabilityLoaded(
                snapshot,
                request_id=request_id,
                purpose=purpose,
            )
        )

    def _is_current_availability_request(self, request_id: int | None) -> bool:
        if self._selected_model is None:
            return False
        if request_id is None:
            return True
        if self._phase == "models":
            return False
        return request_id == self._availability_request_id

    def on_fast_deploy_availability_loaded(
        self,
        message: FastDeployAvailabilityLoaded,
    ) -> None:
        self._snapshot = message.snapshot
        if message.purpose == "filter":
            self._populate_gpu_filter(message.snapshot)
            if self._phase == "models":
                self._render_model_list(preferred_id=self._highlighted_model_id())
            return
        if not self._is_current_availability_request(message.request_id):
            return
        model = self._selected_model
        if model is None:
            self._availability_inflight = False
            return
        self._render_infra_from_snapshot(model, message.snapshot)

    def on_fast_deploy_availability_failed(
        self,
        message: FastDeployAvailabilityFailed,
    ) -> None:
        if message.purpose == "filter":
            return
        if not self._is_current_availability_request(message.request_id):
            return
        model = self._selected_model
        if model is None:
            self._availability_inflight = False
            return
        self._render_fallback(model, reason=message.error)

    def _render_infra_from_snapshot(
        self,
        model: QuickDeployModel,
        snapshot: ComputeAvailabilitySnapshot,
    ) -> None:
        all_rows = infra_rows_for_model(model, snapshot)
        rows = _filter_infra_rows(all_rows, self._gpu_filter)
        if not rows:
            if all_rows and self._gpu_filter not in {"", "any"}:
                self._render_empty_gpu_filter()
                return
            self._render_fallback(
                model,
                reason="No connected placement fits this model's requirements.",
            )
            return
        self._phase = "infra"
        self._availability_inflight = False
        self._infra_rows = {row.plan.quote.id: row for row in rows}
        width_mode = self.viewport_profile.width_mode
        option_list = self.query_one("#fast-deploy-list", OptionList)
        option_list.set_options(
            [
                Option(_infra_option(row, width_mode), id=row.plan.quote.id)
                for row in rows
            ]
        )
        option_list.highlighted = 0
        self._update_infra_detail(rows[0])
        self.query_one("#fast-deploy-subtitle", Static).update(
            f"Pick infrastructure · {_escape_markup(self._catalog_info.source_label)}"
        )
        status = (
            f"[bold]{len(rows)} infrastructure option"
            f"{'s' if len(rows) != 1 else ''}[/bold] "
            "[dim]cheapest first · updated just now[/dim]"
        )
        if self._gpu_filter not in {"", "any"}:
            status += f" [dim]· GPU {_escape_markup(self._gpu_filter)}[/dim]"
        if snapshot.errors:
            status += "\n[yellow]Partial results:[/yellow] " + _escape_markup(
                "; ".join(snapshot.errors)
            )
        self.query_one("#fast-deploy-status", Static).update(status)
        option_list.focus()

    def _render_empty_gpu_filter(self) -> None:
        self._phase = "infra"
        self._availability_inflight = False
        self._infra_rows = {}
        gpu = _escape_markup(self._gpu_filter)
        option_list = self.query_one("#fast-deploy-list", OptionList)
        option_list.set_options(
            [Option(f"  No placements on {gpu}", disabled=True)]
        )
        self.query_one("#fast-deploy-detail", Static).update("")
        self.query_one("#fast-deploy-subtitle", Static).update(
            f"Pick infrastructure · {_escape_markup(self._catalog_info.source_label)}"
        )
        self.query_one("#fast-deploy-status", Static).update(
            f"[yellow]No live placements on {gpu}.[/yellow] "
            "[dim]Press g to change the GPU filter.[/dim]"
        )
        option_list.focus()

    def _render_fallback(self, model: QuickDeployModel, *, reason: str | None) -> None:
        self._phase = "fallback"
        self._availability_inflight = False
        profiles = model.profiles
        connected_providers = self._snapshot.providers if self._snapshot is not None else None
        if connected_providers is not None and ComputeProvider.MODAL not in connected_providers:
            self._fallback_profiles = {}
            option_list = self.query_one("#fast-deploy-list", OptionList)
            option_list.set_options(
                [Option("  No compatible connected provider is available", disabled=True)]
            )
            self.query_one("#fast-deploy-detail", Static).update("")
            status = (
                "[yellow]No compatible connected provider is available.[/yellow]\n"
                "[dim]Catalog fallback estimates require Modal.[/dim]"
            )
            if reason:
                status += f"\n[dim]{_escape_markup(_clip(reason, 120))}[/dim]"
            self.query_one("#fast-deploy-status", Static).update(status)
            return
        self._fallback_profiles = {profile.id: profile for profile in profiles}
        option_list = self.query_one("#fast-deploy-list", OptionList)
        options = [Option(_fallback_option(profile), id=profile.id) for profile in profiles]
        self.query_one("#fast-deploy-subtitle", Static).update(
            f"Catalog estimates · {_escape_markup(self._catalog_info.source_label)}"
        )
        if not options:
            options = [Option("  No catalog profiles for this model", disabled=True)]
        option_list.set_options(options)
        if profiles:
            option_list.highlighted = 0
            self._update_fallback_detail(profiles[0])
        status = "[yellow]Live availability unavailable — showing catalog estimates.[/yellow]"
        if reason:
            status += f"\n[dim]{_escape_markup(_clip(reason, 120))}[/dim]"
        self.query_one("#fast-deploy-status", Static).update(status)

    def _highlighted_model_id(self) -> str | None:
        try:
            option_list = self.query_one("#fast-deploy-list", OptionList)
        except Exception:
            return None
        highlighted = option_list.highlighted_option
        if highlighted is None or highlighted.id is None:
            return None
        return str(highlighted.id)

    def refresh_quick_deploy_catalog(self) -> None:
        """Apply a catalog activation broadcast by the app."""
        self._apply_model_catalog(force=False)

    def _apply_model_catalog(self, *, force: bool = False) -> None:
        if self._phase != "models":
            return
        info = get_quick_deploy_catalog_info()
        models = list_quick_deploy_models()
        if not force and info == self._catalog_info and models == self._models:
            return
        preferred_id = self._highlighted_model_id()
        self._catalog_info = info
        self._models = models
        self._model_by_id = {model.id: model for model in models}
        self._render_model_list(preferred_id=preferred_id)

    def _visible_models(self) -> tuple[QuickDeployModel, ...]:
        if self._gpu_filter == "any" or self._snapshot is None:
            return self._models
        return tuple(
            model
            for model in self._models
            if _model_fits_gpu_type(model, self._snapshot, self._gpu_filter)
        )

    def viewport_profile_changed(
        self,
        profile: ViewportProfile,
        previous: ViewportProfile | None,
    ) -> None:
        """Reformat list records when their available width class changes."""
        if previous is not None and previous.width_mode == profile.width_mode:
            return
        try:
            if self._phase == "models":
                self._render_model_list(preferred_id=self._highlighted_model_id())
            elif (
                self._phase == "infra"
                and self._selected_model is not None
                and self._snapshot is not None
            ):
                self._render_infra_from_snapshot(self._selected_model, self._snapshot)
            elif self._phase == "fallback" and self._selected_model is not None:
                self._render_fallback(self._selected_model, reason=None)
        except Exception:
            return

    def _render_model_list(self, *, preferred_id: str | None = None) -> None:
        option_list = self.query_one("#fast-deploy-list", OptionList)
        visible = self._visible_models()
        options = [
            Option(
                _model_option(
                    model,
                    self.viewport_profile.width_mode,
                    snapshot=self._snapshot,
                ),
                id=model.id,
            )
            for model in visible
        ]
        if not options:
            if self._gpu_filter != "any":
                options = [
                    Option(
                        f"  No catalog models fit {_escape_markup(self._gpu_filter)}",
                        disabled=True,
                    )
                ]
            else:
                options = [Option("  No models in the quick-deploy catalog", disabled=True)]
        option_list.set_options(options)
        highlight_index = 0
        if preferred_id is not None:
            for index, model in enumerate(visible):
                if model.id == preferred_id:
                    highlight_index = index
                    break
        if visible:
            option_list.highlighted = highlight_index
            self._update_model_detail(visible[highlight_index])
        else:
            self.query_one("#fast-deploy-detail", Static).update("")
        self.query_one("#fast-deploy-title", Static).update(
            "[bold #7bf168]Deploy[/]  [dim]Step 1: Pick a model[/dim]"
        )
        self.query_one("#fast-deploy-subtitle", Static).update(_subtitle(self._catalog_info))
        plural = "s" if len(visible) != 1 else ""
        filter_note = ""
        if self._gpu_filter != "any":
            filter_note = f" · GPU {_escape_markup(self._gpu_filter)}"
        self.query_one("#fast-deploy-status", Static).update(
            f"[dim]{len(visible)} model{plural}{filter_note} · "
            f"{_escape_markup(self._catalog_info.source_label)}[/dim]"
        )
        if getattr(self.focused, "id", "") != "fast-deploy-gpu-filter":
            option_list.focus()

    def _populate_gpu_filter(self, snapshot: ComputeAvailabilitySnapshot) -> None:
        selector = self.query_one("#fast-deploy-gpu-filter", Select)
        options = _gpu_filter_options(snapshot)
        current = self._gpu_filter
        values = {value for _label, value in options}
        if current not in values:
            current = "any"
            self._gpu_filter = "any"
        self._updating_gpu_filter = True
        selector.set_options(options)
        selector.value = current
        self._updating_gpu_filter = False

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "fast-deploy-gpu-filter" or self._updating_gpu_filter:
            return
        if not isinstance(event.value, str) or not event.value:
            return
        if event.value == self._gpu_filter:
            return
        self._gpu_filter = event.value
        if self._phase == "models":
            self._render_model_list(preferred_id=self._highlighted_model_id())
            return
        if self._phase == "infra" and self._selected_model is not None and self._snapshot is not None:
            self._render_infra_from_snapshot(self._selected_model, self._snapshot)

    def _reset_to_models(self) -> None:
        self._phase = "models"
        self._selected_model = None
        self._infra_rows = {}
        self._fallback_profiles = {}
        self._availability_inflight = False
        self._apply_model_catalog(force=True)

    def _cancel_availability_request(self) -> None:
        self._availability_inflight = False
        self._availability_request_id += 1

    def _update_model_detail(self, model: QuickDeployModel) -> None:
        self.query_one("#fast-deploy-detail", Static).update(_model_detail(model))

    def _update_infra_detail(self, row: _InfraRow) -> None:
        self.query_one("#fast-deploy-detail", Static).update(_infra_detail(row))

    def _update_fallback_detail(self, profile: QuickDeployProfile) -> None:
        self.query_one("#fast-deploy-detail", Static).update(_fallback_detail(profile))

    def action_choose_selected(self) -> None:
        if isinstance(self.focused, Select):
            raise SkipAction()
        highlighted = self.query_one("#fast-deploy-list", OptionList).highlighted_option
        if highlighted is not None and highlighted.id is not None:
            self._choose(str(highlighted.id))

    def action_focus_gpu_filter(self) -> None:
        self.query_one("#fast-deploy-gpu-filter", Select).focus()

    def action_pop_screen(self) -> None:
        if self._phase == "models":
            self.app.pop_screen()
            return
        self._cancel_availability_request()
        self._reset_to_models()
