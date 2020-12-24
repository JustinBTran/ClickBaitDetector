const puppeteer = require('puppeteer');
const fs = require('fs');

const config = require('./config.json')
const cookies = require('./cookies.json')

LogIn();

async function LogIn(){
    let browser = await puppeteer.launch({headless : false});
    const context  = browser.defaultBrowserContext();

    context.overridePermissions("https://www.facebook.com", []);

    let page = await browser.newPage();
    await page.setDefaultNavigationTimeout(100000);
    await page.setViewport({ width: 1200, height: 800 });

    if (!Object.keys(cookies).length) {
        await page.goto("https://www.facebook.com/login", { waitUntil: "networkidle2" });
        await page.type("#email", config.username, { delay: 10 })
        await page.type("#pass", config.password, { delay: 10 })
        await page.click("#loginbutton");
        await page.waitForNavigation({ waitUntil: "networkidle0" });
        await page.waitFor(15000);
        try {
            console.log("Confirming login");
            await page.waitForSelector('.buofh1pr');
        }catch (err) {
            console.log("failed to login");
            process.exit(0);
        }
        let currentCookies = await page.cookies();
        fs.writeFileSync('./cookies.json', JSON.stringify(currentCookies));
        
    } else{
        //User Already Logged In
        await page.setCookie(...cookies);
        await page.goto("https://www.facebook.com/", { waitUntil: "networkidle2" });
    }

    await GetHeadlines(page);
    await browser.close();

};

async function GetHeadlines(page){
    console.log("Geting Headlines.");
    try{
        var headlines = await page.evaluate(() => {
            const divNamePostings = "[data-pagelet^='FeedUnit']";
            var hl = [];
            var linkedPosts = document.body.querySelectorAll(divNamePostings);
            for(var i =0; i<linkedPosts.length;i++){
                let headline = linkedPosts[i].querySelector("[class^='a8c37x1j ni8dbmo4 stjgntxs l9j0dhe7 ojkyduve']");
                if(headline != null)
                    hl.push(headline.textContent);
            }
            return hl;
        })
        headlines.forEach(element => console.log(element));
    }catch(err){
        console.log(err.message);
    }

};


