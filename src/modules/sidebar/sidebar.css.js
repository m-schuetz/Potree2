
export const css = `


tr:hover {background-color: coral;}


#potree_sidebar{
	background: #333333;
	color: #ffffff;
	font-family: Calibri;
	overflow: hidden;
}

#potree_sidebar_section_selection{
	background: #333333;
	margin: 0 auto;
	padding: 5px 0 0 0;
	display: grid;
	grid-template-rows: repeat(10, 48px);
}

#potree_sidebar_main{
	background: #252525;
	padding: 0.5em;
}

#potree_sidebar_content{

}

#potree_sidebar_footer{
	opacity: 0.6;
}

.potree_sidebar_section_button{
	width: 32px;
	height: 32px;
	background-size: contain;
	background-repeat: no-repeat;
	background-color: transparent;
	border: none;
	margin: 4px 0px;
}

.potree_sidebar_section_button:hover{
	filter: brightness(120%);
	filter: drop-shadow(0px 0px 3px white) drop-shadow(0px 0px 3px white);
}

.potree_sidebar_button{
	width: 3em;
	height: 3em;
	background-size: contain;
	background-repeat: no-repeat;
	background-color: transparent;
	border: none;
}

.potree_sidebar_button:hover{
	filter: brightness(120%);
	filter: drop-shadow(0px 0px 3px white) drop-shadow(0px 0px 3px white);
}

select{
	width: 100%;
}

.subsection_panel{
	margin: 0px 0px 25px 0px;
}

.subsection{
	text-align:        center;
	font-family:       Calibri;
	font-size:         1.1em;
	font-weight:       bold;
	padding:           0.1em 0em 0.5em 0em;
	letter-spacing:    1px;
	z-index:           1;
	overflow:          hidden;
	width:             100%;
	display:           flex;
	gap:               10px;
	text-wrap:         nowrap;
}
.subsection:before, .subsection:after {
	position:          relative;
	top:               51%;
	overflow:          hidden;
	width:             100%;
	height:            1px;
	content:           '-';
	background-color:  #888888;
	align-self:        center;
}








.potree_gradient_button{
	width: 100%;
	height: 3em;
}

 .potree_gradient_button:hover {
	filter: brightness(120%);
}

table{
	color: white;
}

.sidebarlabel{
	white-space: nowrap
}

sidebarlabel{
	white-space: nowrap
}








section.range-slider {
	position: relative;
	user-select: none;
}

section.range-slider input {
	position: absolute;
	pointer-events: none;
	outline: none;
	-webkit-appearance: none;
	background: none;
	width: 100%;
}

section.range-slider input::-webkit-slider-thumb {
	pointer-events: all;
	position: relative;
	z-index: 1;
	outline: 0;
}

.range-slider-background{
	position: absolute;
	background: lightgray;
	width: 100%;
	height: 0.5em;
	border-radius: 0.5em;
	top: calc(50% - 0.25em);
}

.range-slider-selected{
	position: absolute;
	background: red;
	width: 50%;
	height: 0.5em;
	border-radius: 0.5em;
	top: calc(50% - 0.25em);
}

`;