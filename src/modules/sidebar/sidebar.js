
import {Gradients, Utils} from "potree";
import {createAttributesPanel} from "./panel_attributes.js";
import {createMeasurementsPanel} from "./panel_measurements.js";
import {createPanel as createAppearancePanel} from "./panel_appearance.js";
import {createPanel as createInfosPanel} from "./panel_infos.js";

let sidebar = null;
let dir = new URL(import.meta.url + "/../").href;

async function installMainSection(){
	let elButton = document.createElement("input");
	elButton.classList.add("potree_sidebar_section_button");
	elButton.type = "button";
	elButton.title = "Main";
	elButton.style.backgroundImage = `url(${dir}/icons/distance.svg)`;

	elButton.addEventListener("click", () => {
		potree.measure.startMeasuring({
			
		});
	});

	sidebar.elSectionSelection.append(elButton);

	let elMain = document.createElement("span");

	elMain.innerHTML = `
		<div id="attributes_panel">
			
		</div>
	`;
	sidebar.elSectionContent.append(elMain);

	let panel_appearance = createAppearancePanel();
	elMain.append(panel_appearance.element);

	let panel_attributes = createAttributesPanel();
	elMain.append(panel_attributes.element);

	let panel_measurements = createMeasurementsPanel();
	elMain.append(panel_measurements.element);

	let panel_infos = createInfosPanel();
	elMain.append(panel_infos.element);

	
	
}

export async function installSidebar(elPotree, potree){

	let {css} = await import("./sidebar.css.js");

	let style = document.createElement('style');
	style.innerHTML = css;
	document.getElementsByTagName('head')[0].appendChild(style);

	let elSidebar = document.createElement("span");
	elSidebar.id = "potree_sidebar";
	elSidebar.style.display = "grid";
	elSidebar.style.gridTemplateColumns = "3em 1fr";

	elSidebar.innerHTML = `
		<span id="potree_sidebar_section_selection"></span>
		<span id="potree_sidebar_section_content"></span>
	`;

	elPotree.style.display = "grid";
	elPotree.style.gridTemplateColumns = "23em 1fr";
	elPotree.prepend(elSidebar);

	let elSectionSelection = elSidebar.querySelector("#potree_sidebar_section_selection");
	let elSectionContent = elSidebar.querySelector("#potree_sidebar_section_content");

	sidebar = {
		elCointainer: elPotree,
		potree,
		elSidebar, elSectionSelection, elSectionContent
	};

	installMainSection();
}


