import { expect, test } from "@playwright/test";
import path from "node:path";

test("real cached cross-modal search, preview, dimension, and research view", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Find the signal." }),
  ).toBeVisible();
  await expect(page.getByText("14 segments indexed")).toBeVisible();
  await page.screenshot({
    path: "../docs/screenshots/search-home.png",
    fullPage: true,
  });
  await page.getByLabel("Search your knowledge").press("Enter");
  await expect(page.locator(".result-card")).toHaveCount(14);
  await expect(page.locator(".cache-label")).toContainText("Cached embedding");
  await expect(page.locator(".result-card").first()).toContainText(
    /ocean|sailboat/,
  );
  await page.screenshot({
    path: "../docs/screenshots/search-results.png",
    fullPage: true,
  });
  await page
    .locator(".result-card")
    .filter({ hasText: "sailboat.mp4" })
    .click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(
    page.getByText("00:00 – 00:04 · coarse segment match"),
  ).toBeVisible();
  await expect(page.locator(".preview-media video")).toHaveAttribute(
    "src",
    /#t=0,4/,
  );
  await page.getByLabel("Close preview").click();
  await page.getByLabel("Embedding dimension").selectOption("128");
  await page.getByLabel("Search your knowledge").press("Enter");
  await expect(page.locator(".results-meta")).toContainText("128d");
  await page
    .getByRole("button", { name: "Embedding lab", exact: false })
    .click();
  await expect(
    page.getByRole("heading", { name: "Less space. More insight." }),
  ).toBeVisible();
  await expect(page.locator(".experiment-table tbody tr")).toHaveCount(4);
  await page.screenshot({
    path: "../docs/screenshots/embedding-lab.png",
    fullPage: true,
  });
});

test("collection creation, upload, preview and cleanup through the API-connected UI", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Collections", exact: true }).click();
  await page
    .getByRole("button", { name: "New collection", exact: true })
    .last()
    .click();
  await page.getByLabel("Collection name").fill("Browser integration test");
  await page
    .getByLabel("Description", { exact: true })
    .fill("Disposable UI test collection");
  await page
    .getByRole("button", { name: "Create collection", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Browser integration test", exact: true }),
  ).toBeVisible();
  const chooser = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: "Upload files", exact: true }).click();
  await (await chooser).setFiles(path.resolve("../assets/demo/aurora.md"));
  await expect(page.locator(".filename")).toHaveText("aurora.md");
  await page.locator(".filename").click();
  await expect(page.locator(".preview-media pre")).toContainText(
    "Solar particles",
  );
  await page.getByLabel("Close preview").click();
  page.once("dialog", (dialog) => dialog.accept());
  await page
    .getByLabel("Delete Browser integration test", { exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Browser integration test", exact: true }),
  ).toHaveCount(0);
});

test("all navigation views and original audio/image previews", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByRole("button", { name: "Media explorer", exact: true })
    .click();
  await expect(page.locator(".media-tile")).toHaveCount(12);
  await page.getByRole("button", { name: /bell.wav/ }).click();
  await expect(page.locator(".preview-media audio")).toBeVisible();
  await page.getByLabel("Close preview").click();
  await page.getByRole("button", { name: /door.png/ }).click();
  await expect(page.locator(".preview-media img")).toBeVisible();
  await page.getByLabel("Close preview").click();
  await page
    .getByRole("button", { name: "Indexing jobs", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Work in motion." }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "System status", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Built to stay lightweight." }),
  ).toBeVisible();
  await expect(
    page.getByText("google/embeddinggemma-2", { exact: true }),
  ).toBeVisible();
});

test("mobile search remains usable without horizontal page overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByLabel("Search your knowledge").press("Enter");
  await expect(page.locator(".result-card")).toHaveCount(14);
  const dimensions = await page.evaluate(() => ({
    width: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  expect(dimensions.scroll).toBeLessThanOrEqual(dimensions.width);
  await page.screenshot({
    path: "../docs/screenshots/mobile-search.png",
    fullPage: true,
  });
});
