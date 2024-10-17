const examples = [
    {title: "Institut für Arbeitsmarkt- und Berufsforschung (IAB) – unser Leitbild",
    text: `Das IAB schafft wissenschaftliche Grundlagen für fundierte Entscheidungen in der Arbeitsmarktpolitik. Wir erforschen den Arbeitsmarkt und die Berufswelt, um die Debatten in der Wissenschaft, der Öffentlichkeit, der Politik und der Verwaltung voranzubringen. Aus öffentlichen Mitteln finanziert, sind wir dem Gemeinwohl verpflichtet.

So tragen wir zu einer nachhaltigen Entwicklung in Wirtschaft und Gesellschaft bei – mit dem Ziel, die Lebens- und Arbeitsbedingungen der Menschen stetig zu verbessern.

Gleichzeitig leitet uns das Ziel der wissenschaftlichen Exzellenz. Wir streben nach neuen Erkenntnissen, reagieren auf gesellschaftliche Veränderungen, erschließen unterschiedlichste Datenquellen und entwickeln Forschungsansätze für unsere Themen.

Wir arbeiten auf empirischer Basis, ergebnisoffen und wissenschaftlich unabhängig.

Methodenpluralität, Interdisziplinarität und Vernetzung sind für uns selbstverständlich. Wir beteiligen uns intensiv am wissenschaftlichen Diskurs im In- und Ausland. Daten und Forschungsergebnisse teilen wir mit der Wissenschaftsgemeinschaft und der Öffentlichkeit. Dabei ist der Datenschutz stets umfassend gewährleistet.

Durch Nachwuchsförderung und vielfältige Qualifizierungsangebote bieten wir unseren Beschäftigten zahlreiche Möglichkeiten, sich weiterzuentwickeln. In unserem Handeln achten wir auf Transparenz, Partizipation, einen wertschätzenden Umgang miteinander, die Vereinbarkeit von Beruf und Familie sowie auf Nachhaltigkeit. Wir stehen für Chancengleichheit, Inklusion und Diversität.

Source: https://www.iab.de
            `},

            {title: "Geschichte der Stadt Nürnberg",
            text: `Die Geschichte der Stadt Nürnberg setzt mit der ersten urkundlichen Erwähnung 1050 ein. Nürnberg stieg im Mittelalter unter den Staufern und Luxemburgern zu einer der wichtigen Reichsstädte im Heiligen Römischen Reich auf. Dank des blühenden Fernhandels und Handwerks wurde Nürnberg im 15. und 16. Jahrhundert eines der bedeutendsten kulturellen Zentren der Renaissance nördlich der Alpen sowie des Humanismus und der Reformation.

Nach dem Dreißigjährigen Krieg (1618–1648) verlor die Stadt ihre herausragende Stellung mit der Verlagerung der politischen Gewichte im Heiligen Römischen Reich. Die Stadt und ihr Territorium blieben weiter selbstständig und konnten von Handel und Handwerk profitieren. Nürnberg wurde 1806 nach der Auflösung des Alten Reichs in das neugegründete Königreich Bayern eingegliedert. Infolge der Industrialisierung erstarkte die Wirtschaft der Stadt wieder. Zugleich sahen in dieser Zeit Anhänger der Romantik und des Historismus im spätmittelalterlichen Stadtbild ihr Ideal verwirklicht.

Source: https://www.wikipedia.com
            `},

            {title: "Arbeitsmarkt im Strukturwandel",
            text: `Wirtschaft und Arbeitsmarkt unterliegen einem permanenten Wandel. Dieser Wandel wird nicht nur durch langfristige Entwicklungen wie demografische Veränderungen oder Globalisierung beeinflusst, sondern auch durch die digitale Transformation, die Auswirkungen des Klimawandels und unerwartete Ereignissen wie die Covid-19-Pandemie.

Das IAB analysiert und prognostiziert, inwiefern diese Faktoren den Wandel des Arbeitsmarkts beeinflussen, indem es deren Effekte auf Beschäftigung und Arbeitslosigkeit sowie auf Fachkräftebedarfe, Entlohnung, Weiterbildung, Arbeitsbedingungen oder Arbeitsangebot betrachtet. Die berufliche Ebene steht im besonderen Fokus der Analysen, da der strukturelle Wandel am Arbeitsmarkt sich vor allem in Veränderungen der beruflichen Anforderungen, Kompetenzen und Tätigkeiten niederschlägt. Des Weiteren wird analysiert, wie sich der technologische Wandel regional differenziert auswirkt.

Source: https://www.iab.de
`}
];

const load_example = function(nr) {
    var title = document.getElementById("app-input-title");
    var text = document.getElementById("app-input-text");
    var example = examples[nr-1];

    title.value = example.title;
    text.value = example.text;
}

const predict = async function(){
    var result = document.getElementById("app-results");
    result.innerHTML = "Wird berechnet ...";

    var body = {
        title: document.getElementById("app-input-title").value,
        text: document.getElementById("app-input-text").value
    };

    var predictions = await fetch('/api/predict', {
        method: 'POST',
        headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
        body: JSON.stringify(body)
    })
    .then(response => response.json())
    .catch(error => {
        console.error(error);
        return {};
    });


   result.innerHTML = "";
   predictions.forEach(x => {
    result.innerHTML += "<span class='app-result-item'>"+x.word+" <span class='app-result-item-score'>"+Math.round(x.score*10000,2)/100+"</span></span>";
   });
};
