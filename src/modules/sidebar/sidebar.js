
import {Gradients, Utils} from "potree";
import {createAttributesPanel} from "./panel_attributes.js";
import {createMeasurementsPanel} from "./panel_measurements.js";
import {createPanel as createAppearancePanel} from "./panel_appearance.js";
import {createPanel as createInfosPanel} from "./panel_infos.js";
import {createPanel as createHoveredPanel} from "./panel_hovered.js";
import {createPanel as createScenePanel} from "./panel_scene.js";

let sidebar = null;
let dir = new URL(import.meta.url + "/../").href;

class Section{

	constructor(){
		this.icon = null;
		this.panel = null;
	}

}

let sections = [];
let activeSection = null;

function setActiveSection(section){

	if(!section){
		sidebar.elSectionContent.innerHTML = "";
	}else if(section === activeSection){
		toggle();
	}else{
		open();

		sidebar.elSectionContent.innerHTML = "";
		sidebar.elSectionContent.append(section.panel);
	}

	activeSection = section;
}

let isOpen = true;
function toggle(){
	if(isOpen){
		isOpen = false;
		sidebar.elContainer.style.gridTemplateColumns = "48px 1fr";
	}else{
		isOpen = true;
		sidebar.elContainer.style.gridTemplateColumns = "23em 1fr";
	}
}

function open(){
	if(!isOpen){
		isOpen = true;
		sidebar.elContainer.style.gridTemplateColumns = "23em 1fr";
	}
}

function onSectionSelect(section){

	setActiveSection(section);
}

function addSection(section){

	{
		let elButton = document.createElement("input");
		elButton.classList.add("potree_sidebar_section_button");
		elButton.type = "button";
		elButton.title = "Measure";
		elButton.style.backgroundImage = section.icon;

		elButton.addEventListener("click", () => {
			onSectionSelect(section);
		});

		sidebar.elSectionSelection.append(elButton);
	}


	sections.push(section);
}

function createMainSection(){

	let elPanel = document.createElement("span");

	elPanel.innerHTML = `
		<div id="attributes_panel">
			
		</div>
	`;

	let panel_appearance = createAppearancePanel();
	elPanel.append(panel_appearance.element);

	let panel_scene = createScenePanel();
	elPanel.append(panel_scene.element);

	let panel_infos = createInfosPanel();
	elPanel.append(panel_infos.element);

	let panel_hovered = createHoveredPanel();
	elPanel.append(panel_hovered.element);

	let section = new Section();
	section.icon = `url(${dir}/icons/home.svg)`;
	section.panel = elPanel;

	return section;
}

function createAttributesSection(){

	let elPanel = document.createElement("span");

	elPanel.innerHTML = `
		<div id="attributes_panel">
			
		</div>
	`;

	let panel_attributes = createAttributesPanel();
	elPanel.append(panel_attributes.element);

	let section = new Section();
	section.icon = `url(${dir}/icons/material.svg)`;
	section.panel = elPanel;

	return section;
}

function createMeasureSection(){

	let elPanel = document.createElement("span");

	elPanel.innerHTML = `
		<div id="attributes_panel">
			
		</div>
	`;

	let panel_measurements = createMeasurementsPanel();
	elPanel.append(panel_measurements.element);

	let section = new Section();
	section.icon = `url(${dir}/icons/measure.svg)`;
	section.panel = elPanel;

	return section;
}


export async function installSidebar(elPotree, potree){

	let {css} = await import("./sidebar.css.js");

	let style = document.createElement('style');
	style.innerHTML = css;
	document.getElementsByTagName('head')[0].appendChild(style);

	let elSidebar = document.createElement("span");
	elSidebar.id = "potree_sidebar";
	elSidebar.style.display = "grid";
	elSidebar.style.gridTemplateColumns = "48px 1fr";

	elSidebar.innerHTML = `
		<span id="potree_sidebar_section_selection"></span>
		<span id="potree_sidebar_main" style="display: flex; flex-direction: column;">
			<span id="potree_sidebar_content"></span>
			<!--
			<span style="flex-grow: 100;"></span>
			<span id="potree_sidebar_footer">
				Potree ${Potree.version}<br>
				<a href="https://github.com/m-schuetz/Potree2" target="_blank">github</a>
			</span>
			-->
		</span>
	`;

	elPotree.style.display = "grid";
	elPotree.style.gridTemplateColumns = "23em 1fr";
	elPotree.prepend(elSidebar);

	let elSectionSelection = elSidebar.querySelector("#potree_sidebar_section_selection");
	let elSectionContent = elSidebar.querySelector("#potree_sidebar_content");

	sidebar = {
		elContainer: elPotree,
		potree,
		elSidebar, elSectionSelection, elSectionContent
	};


	let secMain = createMainSection(potree);
	let secMeasure = createMeasureSection();
	let secAttributes = createAttributesSection();

	addSection(secMain);
	// addSection(secAttributes);
	// addSection(secMeasure);

	setActiveSection(secMain);

}


